-- | Option "Greeks": the sensitivities of a Black-Scholes European call
-- price to each of its five inputs, for a portfolio of options.
--
-- This is the control for 'reaction-network.fut' and 'batch-reactor.fut':
-- it satisfies the same cost-model conditions they do, but its primal is a
-- straight-line expression rather than a sequential loop, so the compiler
-- can hoist the primal out of the seed map by itself.  It is the shape of
-- program vector AD should be good at, for two reasons.
--
-- * The primal is dominated by transcendentals -- 'log', 'sqrt', 'exp'
--   and two error functions -- while the derivative of each of them is a
--   multiplication by a factor that the primal has already computed (or
--   that is computed once and shared).  The per-seed tangent is a handful
--   of multiply-adds, so the primal is worth a lot more than the tangent.
--
-- * Everything the tangent touches is a scalar, so no tangent ever has to
--   be materialised as an array.
--
-- Note that the output is scalar, so reverse mode gets the whole gradient
-- in a single sweep and its vector-AD entry has a seed width of one.

-- ==
-- entry: calculate_objective
-- script input { mk_opts 1000000i64 } output @ data/opts_1000000.F

-- ==
-- entry: calculate_jacobian_fwd
-- script input { mk_opts 1000000i64 } output @ data/opts_1000000.J

-- ==
-- entry: calculate_jacobian_fwd_vec
-- script input { mk_opts 1000000i64 } output @ data/opts_1000000.J

-- ==
-- entry: calculate_jacobian_rev
-- script input { mk_opts 1000000i64 } output @ data/opts_1000000.J

-- ==
-- entry: calculate_jacobian_rev_vec
-- script input { mk_opts 1000000i64 } output @ data/opts_1000000.J

def normcdf (x: f64) : f64 =
  0.5 * (1 + f64.erf (x / f64.sqrt 2))

def black_scholes (s: f64, k: f64, r: f64, v: f64, t: f64) : f64 =
  let sqrt_t = f64.sqrt t
  let d1 = (f64.log (s / k) + (r + 0.5 * v * v) * t) / (v * sqrt_t)
  let d2 = d1 - v * sqrt_t
  in s * normcdf d1 - k * f64.exp (-r * t) * normcdf d2

type option = (f64, f64, f64, f64, f64)

-- One seed per input: spot, strike, rate, volatility, time.
def seeds : [5]option =
  [ (1, 0, 0, 0, 0)
  , (0, 1, 0, 0, 0)
  , (0, 0, 1, 0, 0)
  , (0, 0, 0, 1, 0)
  , (0, 0, 0, 0, 1)
  ]

def unpack (o: [5]f64) : option = (o[0], o[1], o[2], o[3], o[4])

def pack ((s, k, r, v, t): option) : [5]f64 = [s, k, r, v, t]

entry calculate_objective [n] (opts: [n][5]f64) : [n]f64 =
  map (black_scholes <-< unpack) opts

-- All four Jacobian entries produce the same five sensitivities per option.

-- | One 'jvp' per input: the primal is evaluated five times per option.
entry calculate_jacobian_fwd [n] (opts: [n][5]f64) : [n][5]f64 =
  map (\o -> map (jvp black_scholes (unpack o)) seeds) opts

-- | One 'jmp' over all five inputs: the primal is evaluated once.
entry calculate_jacobian_fwd_vec [n] (opts: [n][5]f64) : [n][5]f64 =
  map (\o -> jmp black_scholes (unpack o) seeds) opts

-- | The output is scalar, so a single 'vjp' suffices.
entry calculate_jacobian_rev [n] (opts: [n][5]f64) : [n][5]f64 =
  map (\o -> pack (vjp black_scholes (unpack o) 1)) opts

-- | As above via 'mjp', whose seed vector is necessarily of width one.
entry calculate_jacobian_rev_vec [n] (opts: [n][5]f64) : [n][5]f64 =
  map (\o -> pack (mjp black_scholes (unpack o) [1])[0]) opts

-- | A plausible option portfolio: spot and strike around 100, rates in
-- [1%,10%], volatilities in [10%,50%], maturities in [0.1,2.1].  Used as
-- 'script input' below, so there are no data files to keep in sync.
entry mk_opts (n: i64) : [n][5]f64 =
  tabulate n (\i ->
                let u = f64.i64 (i % 997) / 997
                let w = f64.i64 (i % 389) / 389
                in [ 80 + 40 * u
                   , 80 + 40 * w
                   , 0.01 + 0.09 * u
                   , 0.1 + 0.4 * w
                   , 0.1 + 2 * u
                   ])
