-- | Adjoint sensitivity analysis of a non-isothermal batch reactor.
--
-- A companion to 'reaction-network.fut', but more suitable for reverse-mode.
-- Six reactions run in a vessel whose temperature is itself a state variable:
-- each reaction releases heat, and the vessel loses heat to its surroundings.
-- 20 parameters drive 7 observed quantities, so the Jacobian is much cheaper to
-- build a row at a time -- one 'vjp' per output, the "adjoint method" as it is
-- normally described in kinetics. (Thermal-runaway sensitivity is the usual
-- reason to want it.)
--
-- What vector AD amortises here is the taped forward sweep, which does not
-- depend on the cotangent seed; only the return sweep does. Three properties
-- make that worth doing:
--
-- * The forward sweep is a sequential loop over timesteps, so the compiler
--   cannot hoist it out of the map over seeds. 7 'vjp's really do integrate and
--   tape the reactor 7 times.
--
-- * Because the rate constants depend on the *state* temperature, the six
--   'exp's cannot be hoisted out of the timestep loop either -- they are
--   evaluated twenty-four times per step, which is what makes the forward sweep
--   expensive. Differentiating an 'exp' is a multiplication by a value the
--   forward sweep has already taped, so the return sweep is the cheap half.
--   Amortising the expensive half across seeds is the point.
--
-- * State and parameters are scalars, so the adjoints are scalars too --
--   no accumulators, and nothing to materialise as an array.
--
-- Reverse-mode vector AD has a large fixed cost but a very small marginal
-- cost per seed, so it needs a decent number of outputs before it pays.
--
-- Both modes are provided, with and without vector AD. The Jacobian is 7x20, so
-- reverse needs seven sweeps where forward needs twenty.

-- ==
-- entry: calculate_objective
-- script input { mk_reactors 15000i64 } output @ data/reactors4_15000.F

-- ==
-- entry: calculate_jacobian_rev
-- script input { mk_reactors 15000i64 } output @ data/reactors4_15000.J

-- ==
-- entry: calculate_jacobian_rev_vec
-- script input { mk_reactors 15000i64 } output @ data/reactors4_15000.J

-- ==
-- entry: calculate_jacobian_fwd
-- script input { mk_reactors 15000i64 } output @ data/reactors4_15000.J

-- ==
-- entry: calculate_jacobian_fwd_vec
-- script input { mk_reactors 15000i64 } output @ data/reactors4_15000.J

type params = (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)
type state = (f64, f64, f64, f64, f64, f64, f64)

def gas_constant : f64 = 8.314

-- Arrhenius rate constant, at the reactor's current temperature.
def rate (a: f64) (e: f64) (temp: f64) : f64 =
  a * f64.exp (-e / (gas_constant * temp))

-- A chain c1 -> c2 -> ... -> c6, plus a c2 + c3 shortcut, each step
-- releasing heat; the vessel cools towards ambient.
def deriv ((a1, e1, a2, e2, a3, e3, a4, e4, a5, e5, a6, e6, dh1, dh2, dh3, dh4, dh5, dh6, cool, tamb): params) ((c1, c2, c3, c4, c5, c6, temp): state) : state =
  let r1 = rate a1 e1 temp * c1
  let r2 = rate a2 e2 temp * c2
  let r3 = rate a3 e3 temp * c3
  let r4 = rate a4 e4 temp * c4
  let r5 = rate a5 e5 temp * c5
  let r6 = rate a6 e6 temp * c2 * c3
  in ( -r1
     , r1 - r2 - r6
     , r2 - r3 - r6
     , r3 - r4
     , r4 - r5
     , r5 + r6 - 0.01 * c6
     , dh1 * r1 + dh2 * r2 + dh3 * r3 + dh4 * r4 + dh5 * r5 + dh6 * r6
       - cool * (temp - tamb)
     )

def add ((c1, c2, c3, c4, c5, c6, temp): state) ((c1', c2', c3', c4', c5', c6', temp'): state) : state =
  (c1 + c1', c2 + c2', c3 + c3', c4 + c4', c5 + c5', c6 + c6', temp + temp')

def scale (s: f64) ((c1, c2, c3, c4, c5, c6, temp): state) : state =
  (s * c1, s * c2, s * c3, s * c4, s * c5, s * c6, s * temp)

def rk4_step (p: params) (h: f64) (s: state) : state =
  let k1 = deriv p s
  let k2 = deriv p (add s (scale (h / 2) k1))
  let k3 = deriv p (add s (scale (h / 2) k2))
  let k4 = deriv p (add s (scale h k3))
  in add s (scale (h / 6) (add k1 (add (scale 2 k2) (add (scale 2 k3) k4))))

def steps : i64 = 200
def dt : f64 = 0.05

def simulate (s0: state) (p: params) : state =
  loop s = s0 for _i < steps do rk4_step p dt s

-- One cotangent seed per observed quantity.
def rev_seeds_lit : [7]state =
  [ (1, 0, 0, 0, 0, 0, 0)
  , (0, 1, 0, 0, 0, 0, 0)
  , (0, 0, 1, 0, 0, 0, 0)
  , (0, 0, 0, 1, 0, 0, 0)
  , (0, 0, 0, 0, 1, 0, 0)
  , (0, 0, 0, 0, 0, 1, 0)
  , (0, 0, 0, 0, 0, 0, 1)
  ]

def unpack_params (r: [27]f64) : params =
  (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19])

def unpack_state (r: [27]f64) : state =
  (r[20], r[21], r[22], r[23], r[24], r[25], r[26])

def pack_state ((c1, c2, c3, c4, c5, c6, temp): state) : [7]f64 = [c1, c2, c3, c4, c5, c6, temp]

def pack_params ((a1, e1, a2, e2, a3, e3, a4, e4, a5, e5, a6, e6, dh1, dh2, dh3, dh4, dh5, dh6, cool, tamb): params) : [20]f64 = [a1, e1, a2, e2, a3, e3, a4, e4, a5, e5, a6, e6, dh1, dh2, dh3, dh4, dh5, dh6, cool, tamb]

def params_of (r: [20]f64) : params =
  ( r[0]
  , r[1]
  , r[2]
  , r[3]
  , r[4]
  , r[5]
  , r[6]
  , r[7]
  , r[8]
  , r[9]
  , r[10]
  , r[11]
  , r[12]
  , r[13]
  , r[14]
  , r[15]
  , r[16]
  , r[17]
  , r[18]
  , r[19]
  )

-- The reverse seeds are the unit cotangents of the 7 observed quantities;
-- the forward seeds perturb each of the 20 parameters in turn.
def rev_seeds : [7]state = rev_seeds_lit

def fwd_seeds : [20]params =
  tabulate 20 (\i -> params_of (tabulate 20 (\j -> f64.bool (i == j))))

-- | Final concentrations and temperature for every reactor.
entry calculate_objective [n] (reactors: [n][27]f64) : [n][7]f64 =
  map (\r -> pack_state (simulate (unpack_state r) (unpack_params r))) reactors

-- All four Jacobian entries produce the same 7x20 block per reactor: one
-- row per observed quantity, one column per parameter.  The forward results
-- are transposed to match, so any two entries can be diffed against each
-- other.

-- | One 'vjp' per observed quantity: the reactor is integrated and taped
-- 7 times.
entry calculate_jacobian_rev [n] (reactors: [n][27]f64) : [n][7][20]f64 =
  map (\r ->
         let f = simulate (unpack_state r)
         in map (pack_params <-< vjp f (unpack_params r)) rev_seeds)
      reactors

-- | One 'mjp' over all 7 cotangents: taped once.
entry calculate_jacobian_rev_vec [n] (reactors: [n][27]f64) : [n][7][20]f64 =
  map (\r ->
         let f = simulate (unpack_state r)
         in map pack_params (mjp f (unpack_params r) rev_seeds))
      reactors

-- | One 'jvp' per parameter: the reactor is integrated 20 times.
entry calculate_jacobian_fwd [n] (reactors: [n][27]f64) : [n][7][20]f64 =
  map (\r ->
         let f = simulate (unpack_state r)
         in transpose (map (pack_state <-< jvp f (unpack_params r)) fwd_seeds))
      reactors

-- | One 'jmp' over all 20 parameters: integrated once.
entry calculate_jacobian_fwd_vec [n] (reactors: [n][27]f64) : [n][7][20]f64 =
  map (\r ->
         let f = simulate (unpack_state r)
         in transpose (map pack_state (jmp f (unpack_params r) fwd_seeds)))
      reactors

-- | A batch of non-isothermal reactors: six Arrhenius pairs, six heats of
-- reaction, a cooling coefficient and ambient temperature, then the initial
-- concentrations and temperature.  Parameters are chosen so that the
-- exotherm is real but bounded -- temperature climbs a few tens of kelvin
-- and the cooling catches it -- rather than running away.  Used as 'script
-- input' below, so there are no data files to keep in sync.
entry mk_reactors (n: i64) : [n][27]f64 =
  tabulate n (\i ->
                let u = f64.i64 (i % 997) / 997
                let w = f64.i64 (i % 389) / 389
                in [ 2.0e9 + 8e8 * u
                   , 6.0e4 + 1e3 * w
                   , 1.2e9 + 8e8 * w
                   , 5.9e4 + 1e3 * u
                   , 8e8 + 4e8 * u
                   , 6.2e4 + 1e3 * w
                   , 1.6e9 + 4e8 * w
                   , 6.0e4 + 1e3 * u
                   , 1.2e9 + 4e8 * u
                   , 6.1e4 + 1e3 * w
                   , 4e8 + 4e8 * w
                   , 6.3e4 + 1e3 * u
                   , 8 + 4 * u
                   , 6 + 3 * w
                   , 5 + 3 * u
                   , 4 + 3 * w
                   , 5 + 2 * u
                   , 7 + 3 * w
                   , 0.5 + 0.5 * w
                   , 300
                   , 1 + u
                   , 0.05 * w
                   , 0.01 * u
                   , 0
                   , 0
                   , 0
                   , 300 + 10 * u
                   ])
