-- | Parameter sensitivities of a chemical reaction network.
--
-- A batch of independent reactors is integrated with RK4, and we want the
-- Jacobian of the final concentrations with respect to the five kinetic
-- parameters and the three initial concentrations -- a 3x8 block per reactor.
-- This is the standard "local sensitivity analysis" computation in kinetics and
-- systems biology.
--
-- Both modes are provided, with and without vector AD.  Note that the
-- Jacobian is 3x8, so reverse mode needs only three sweeps where forward
-- needs eight: this program's own shape argues for reverse, and the
-- forward entries are here to exercise 'jmp', not because they are how one
-- would actually compute this.
--
-- Two properties make this a good fit for vector AD:
--
-- * The primal is a sequential loop over timesteps, so the compiler cannot
--   hoist it out of the map over seeds.
--
-- * The state and parameters are scalars, so the tangent vectors are small and
--   of statically known size.
--
-- The gain is nonetheless modest, because the primal is cheap.  The rate
-- constants depend only on the parameters (the temperature is a parameter,
-- not a state variable), so the compiler hoists both 'exp's out of the
-- timestep loop.  What remains per step is a short chain of arithmetic, which
-- costs about as much as a single tangent, so sharing it cannot give much more
-- than a 2x speedup in forward mode.  Reverse mode gains more because it tapes
-- the forward sweep once rather than three times.  See 'batch-reactor.fut'
-- for a variant where the 'exp's stay inside the loop.
--
-- On the GPU the vector entries are compiled differently from the scalar
-- ones: the timestep loop ends up outside the kernels, with separate primal
-- and tangent kernels per step and the state and tangents kept in global
-- memory, whereas each scalar entry is a single kernel.

type params = (f64, f64, f64, f64, f64)
type state = (f64, f64, f64)

def gas_constant : f64 = 8.314

-- Arrhenius rate constant.
def rate (a: f64) (e: f64) (temp: f64) : f64 =
  a * f64.exp (-e / (gas_constant * temp))

-- X -> Y -> Z, with the second step inhibited by the product.
def deriv ((a1, e1, a2, e2, temp): params) ((x, y, z): state) : state =
  let r1 = rate a1 e1 temp * x
  let r2 = rate a2 e2 temp * y / (1 + z * z)
  in (-r1, r1 - r2, r2)

def add ((x, y, z): state) ((x', y', z'): state) : state =
  (x + x', y + y', z + z')

def scale (c: f64) ((x, y, z): state) : state =
  (c * x, c * y, c * z)

def rk4_step (p: params) (h: f64) (s: state) : state =
  let k1 = deriv p s
  let k2 = deriv p (add s (scale (h / 2) k1))
  let k3 = deriv p (add s (scale (h / 2) k2))
  let k4 = deriv p (add s (scale h k3))
  in add s (scale (h / 6) (add k1 (add (scale 2 k2) (add (scale 2 k3) k4))))

def steps : i64 = 200
def dt : f64 = 0.005

def simulate (s0: state) (p: params) : state =
  loop s = s0 for _i < steps do rk4_step p dt s

-- The forward seeds perturb each of the five parameters and each of the
-- three initial concentrations in turn; the reverse seeds are the unit
-- cotangents of the three final concentrations.
type inputs = (params, state)

def fwd_seeds : [8]inputs =
  [ ((1, 0, 0, 0, 0), (0, 0, 0))
  , ((0, 1, 0, 0, 0), (0, 0, 0))
  , ((0, 0, 1, 0, 0), (0, 0, 0))
  , ((0, 0, 0, 1, 0), (0, 0, 0))
  , ((0, 0, 0, 0, 1), (0, 0, 0))
  , ((0, 0, 0, 0, 0), (1, 0, 0))
  , ((0, 0, 0, 0, 0), (0, 1, 0))
  , ((0, 0, 0, 0, 0), (0, 0, 1))
  ]

def rev_seeds : [3]state = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

def run ((p, s0): inputs) : state = simulate s0 p

def unpack (r: [8]f64) : inputs = ((r[0], r[1], r[2], r[3], r[4]), (r[5], r[6], r[7]))

def pack_state ((x, y, z): state) : [3]f64 = [x, y, z]

def pack_inputs (((a1, e1, a2, e2, t), (x, y, z)): inputs) : [8]f64 =
  [a1, e1, a2, e2, t, x, y, z]

-- | Final concentrations for every reactor.
entry calculate_objective [n] (reactors: [n][8]f64) : [n][3]f64 =
  map (pack_state <-< run <-< unpack) reactors

-- All four Jacobian entries produce the same 3x8 block per reactor: one row
-- per final concentration, one column per input.  The forward results are
-- transposed to match, so any two entries can be diffed against each other.

-- | One 'jvp' per input: the system is integrated eight times.
entry calculate_jacobian_fwd [n] (reactors: [n][8]f64) : [n][3][8]f64 =
  map (\r -> transpose (map (pack_state <-< jvp run (unpack r)) fwd_seeds)) reactors

-- | One 'jmp' over all eight inputs: the system is integrated once.
entry calculate_jacobian_fwd_vec [n] (reactors: [n][8]f64) : [n][3][8]f64 =
  map (\r -> transpose (map pack_state (jmp run (unpack r) fwd_seeds))) reactors

-- | One 'vjp' per final concentration: the system is integrated and taped
-- three times.
entry calculate_jacobian_rev [n] (reactors: [n][8]f64) : [n][3][8]f64 =
  map (\r -> map (pack_inputs <-< vjp run (unpack r)) rev_seeds) reactors

-- | One 'mjp' over all three cotangents: taped once.
entry calculate_jacobian_rev_vec [n] (reactors: [n][8]f64) : [n][3][8]f64 =
  map (\r -> map pack_inputs (mjp run (unpack r) rev_seeds)) reactors

-- | A batch of reactors: pre-exponential factors, activation energies, a
-- temperature, and initial concentrations.  Used as 'script input' below,
-- so there are no data files to keep in sync.
entry mk_reactors (n: i64) : [n][8]f64 =
  tabulate n (\i ->
                let u = f64.i64 (i % 997) / 997
                let w = f64.i64 (i % 389) / 389
                in [ 1e3 + 1e3 * u
                   , 2e4 + 1e4 * w
                   , 5e2 + 1e3 * w
                   , 2.5e4 + 1e4 * u
                   , 300 + 200 * u
                   , 1 + u
                   , 0.1 * w
                   , 0.01 * u
                   ])

-- Workload: 'mk_reactors 200000i64', which puts the
-- 'calculate_jacobian_fwd' baseline in benchmarking range for 'hip'.  All
-- four Jacobian entries compute the same 3x8 block per reactor, so one
-- expected file covers them.

-- ==
-- entry: calculate_objective
-- script input { mk_reactors 200000i64 } output @ data/reactors_200000.F

-- ==
-- entry: calculate_jacobian_fwd
-- script input { mk_reactors 200000i64 } output @ data/reactors_200000.J

-- ==
-- entry: calculate_jacobian_fwd_vec
-- script input { mk_reactors 200000i64 } output @ data/reactors_200000.J

-- ==
-- entry: calculate_jacobian_rev
-- script input { mk_reactors 200000i64 } output @ data/reactors_200000.J

-- ==
-- entry: calculate_jacobian_rev_vec
-- script input { mk_reactors 200000i64 } output @ data/reactors_200000.J
