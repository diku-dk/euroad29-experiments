-- ht computes 26-28 'jvp's (one per parameter), so the full-width 'jmp' has a
-- seed width of 26-28.
--
-- Full width loses for three reasons:
--
-- * The primal is cheap relative to the tangents (a scalar 'jvp' costs about
--   1.5x the objective), so there is little to amortise.
--
-- * The number of seeds depends on whether 'us' is empty, so the tangent
--   width is not a compile-time constant.  Every tangent is a dynamically
--   sized array, and the skinning loops materialise and transpose [M][d]
--   intermediates.
--
-- * The scalar version parallelises over the seeds, each thread computing a
--   whole objective.  The vector version has no seed map to parallelise over,
--   so its parallelism must come from the inner maps inside the sequential
--   loops over bones, which are small.
--
-- On GPU, it can squeak in a win if we autotune to take advantage of
-- incremental flattening.

-- ==
-- entry: calculate_objective
-- compiled input @ data/ht12_complicated_t26_c100000.in

-- ==
-- entry: calculate_jacobian
-- compiled input @ data/ht12_complicated_t26_c100000.in

-- ==
-- entry: calculate_jacobian_vec
-- compiled input @ data/ht12_complicated_t26_c100000.in

import "lib/github.com/diku-dk/linalg/linalg"

module linalg_f64 = mk_linalg f64

def matmul = linalg_f64.matmul
def matadd = map2 (map2 (f64.+))

def identity n = tabulate_2d n n (\i j -> f64.bool (i == j))

def angle_axis_to_rotation_matrix (angle_axis: [3]f64) : [3][3]f64 =
  let n = f64.sqrt (angle_axis[0] ** 2 + angle_axis[1] ** 2 + angle_axis[2] ** 2)
  in if n < 0.0001
     then #[sequential] identity 3
     else let x = angle_axis[0] / n
          let y = angle_axis[1] / n
          let z = angle_axis[2] / n
          let s = f64.sin n
          let c = f64.cos n
          in [ [ x * x + (1 - x * x) * c
               , x * y * (1 - c) - z * s
               , x * z * (1 - c) + y * s
               ]
             , [ x * y * (1 - c) + z * s
               , y * y + (1 - y * y) * c
               , y * z * (1 - c) - x * s
               ]
             , [ x * z * (1 - c) - y * s
               , z * y * (1 - c) + x * s
               , z * z + (1 - z * z) * c
               ]
             ]

def apply_global_transform [n] [m] (pose_params: [n][3]f64) (positions: [3][m]f64) : [3][m]f64 =
  let R =
    angle_axis_to_rotation_matrix pose_params[0]
    |> map (map2 (*) pose_params[1])
  in (R `matmul` positions)
     `matadd` transpose (replicate m [pose_params[2, 0], pose_params[2, 1], pose_params[2, 2]])

def relatives_to_absolutes [n] (relatives: [n][4][4]f64) (parents: [n]i32) : [n][4][4]f64 =
  -- Initial value does not matter.
  loop absolutes: *[n][4][4]f64 = replicate n (identity 4)
  for i < n do
    let relative = relatives[i]
    let parent = parents[i]
    in absolutes with [i] = if parent == -1
                 then relative
                 else (absolutes[parent] `matmul` relative)

def euler_angles_to_rotation_matrix (xzy: [3]f64) : [4][4]f64 =
  let tx = xzy[0]
  let ty = xzy[2]
  let tz = xzy[1]
  let costx = f64.cos (tx)
  let sintx = f64.sin (tx)
  let costy = f64.cos (ty)
  let sinty = f64.sin (ty)
  let costz = f64.cos (tz)
  let sintz = f64.sin (tz)
  in [ [ costy * costz
       , -costx * sintz + sintx * sinty * costz
       , sintx * sintz + costx * sinty * costz
       , 0
       ]
     , [ costy * sintz
       , costx * costz + sintx * sinty * sintz
       , -sintx * costz + costx * sinty * sintz
       , 0
       ]
     , [ -sinty
       , sintx * costy
       , costx * costy
       , 0
       ]
     , [ 0
       , 0
       , 0
       , 1
       ]
     ]

type~ hand_model [num_bones] [M] =
  { parents: [num_bones]i32
  , base_relatives: [num_bones][4][4]f64
  , inverse_base_absolutes: [num_bones][4][4]f64
  , weights: [num_bones][M]f64
  , base_positions: [4][M]f64
  , triangles: [][3]i32
  , is_mirrored: bool
  }

def get_posed_relatives [num_bones] [M] (model: hand_model [num_bones] [M]) (pose_params: [][3]f64) =
  let offset = 3
  let f i =
    matmul model.base_relatives[i]
           (euler_angles_to_rotation_matrix pose_params[i + offset])
  in tabulate num_bones f

def get_skinned_vertex_positions [num_bones] [M]
                                 (model: hand_model [num_bones] [M])
                                 (pose_params: [][3]f64)
                                 (apply_global: bool) =
  let relatives = get_posed_relatives model pose_params
  let absolutes = relatives_to_absolutes relatives model.parents
  let transforms = map2 matmul absolutes model.inverse_base_absolutes
  let base_positions = model.base_positions
  let positions =
    loop pos = tabulate_2d 3 M (\_ _ -> 0)
    for i < num_bones do
      let transform = transforms[i]
      let weights = model.weights[i]
      in map2 (map2 (+))
              pos
              (transform[0:3]
               `matmul` base_positions
                        |> map (map2 (*) weights))
  let positions =
    if model.is_mirrored
    then positions with [0] = map f64.neg positions[0]
    else positions
  in if apply_global
     then apply_global_transform pose_params positions
     else positions

def to_pose_params (theta: []f64) (num_bones: i64) : [][]f64 =
  let n = 3 + num_bones
  let num_fingers = 5
  let cols = 5 + num_fingers * 4
  in tabulate n (\i ->
                   match i
                   case 0 -> take 3 theta[0:]
                   case 1 -> [1, 1, 1]
                   case 2 -> take 3 theta[3:]
                   case j ->
                     if j >= cols || j == 3 || j % 4 == 0
                     then [0, 0, 0]
                     else if j % 4 == 1
                     then [theta[j + 1], theta[j + 2], 0]
                     else [theta[j + 2], 0, 0])

def (+^) = map2 (f64.+)

-- Not sure if this should be a run-time parameter, but it is constant
-- for all datasets, and seems more like an algorithmic property (it's
-- an encoding of various spatial transformations).
def theta_count : i64 = 26

def objective [num_us] [num_bones] [N] [M]
              (model: hand_model [num_bones] [M])
              (correspondences: [N]i32)
              (points: [3][N]f64)
              (theta: [theta_count]f64)
              (us: [num_us]f64) : [N][3]f64 =
  let pose_params = to_pose_params theta num_bones
  let vertex_positions = get_skinned_vertex_positions model pose_params true
  in if length us == 0
     then -- "Simple" case
          map2 (\point correspondence ->
                  map2 (-) point vertex_positions[:, correspondence])
               (transpose points)
               correspondences
     else -- "Complex" case
          let us = unflatten (sized (N * 2) us)
          in map3 (\point correspondence u ->
                     let verts = model.triangles[correspondence]
                     let hand_point =
                       map (* u[0]) (vertex_positions[:, verts[0]])
                       +^ map (* u[1]) vertex_positions[:, verts[1]]
                       +^ map (* (1 - u[0] - u[1])) (vertex_positions[:, verts[2]])
                     in map2 (-) point hand_point)
                  (transpose points)
                  correspondences
                  us

-- All parameters up to and including 'is_mirrored' constitute the
-- model.  Of the remaining three parameters, 'theta' is called 'p' in
-- the paper.
entry calculate_objective [num_bones] [N] [M]
                          (parents: [num_bones]i32)
                          (base_relatives: [num_bones][4][4]f64)
                          (inverse_base_absolutes: [num_bones][4][4]f64)
                          (weights: [num_bones][M]f64)
                          (base_positions: [4][M]f64)
                          (triangles: [][3]i32)
                          (is_mirrored: bool)
                          (correspondences: [N]i32)
                          (points: [3][N]f64)
                          (theta: [theta_count]f64)
                          (us: []f64) : [N][3]f64 =
  let model: hand_model [num_bones] [M] =
    { parents
    , base_relatives
    , inverse_base_absolutes
    , weights
    , base_positions
    , triangles
    , is_mirrored
    }
  in objective model correspondences points theta us

-- | Jacobians.
--
-- The objective is differentiated with respect to 'theta' (26 scalars)
-- and 'us' (two scalars per point).  Since output row 'i' depends only
-- on 'us[i]', all points' 'us[·][0]' can share a seed, and likewise
-- for 'us[·][1]'; hence 'theta_count+2' seed vectors suffice for the
-- 'complicated' case, and 'theta_count' for the 'simple' one.
--
-- Seed 'i' is: one at 'theta[i]', or -- for 'i >= theta_count' -- one
-- at 'us[·][i-theta_count]'.  Seeds beyond 'num_seeds' are entirely
-- zero, which is what makes the chunked version's padding harmless.
def num_seeds (num_us: i64) = theta_count + if num_us == 0 then 0 else 2

def seed (num_us: i64) (i: i64) : ([theta_count]f64, [num_us]f64) =
  ( tabulate theta_count (\j -> f64.bool (i == j))
  , tabulate num_us (\j -> f64.bool (i >= theta_count && j % 2 == i - theta_count))
  )

-- ADBench expects the 'us' derivatives in the first two columns, but
-- they are the last seeds.
def reorder [num_us] (_us: [num_us]f64) (J: [][]f64) =
  if num_us == 0 then J else rotate (-2) J

-- The Jacobian is morally transposed, because that is what ADBench expects.

-- | One 'jvp' per seed: the primal is recomputed 'num_seeds' times.
entry calculate_jacobian [num_bones] [N] [M] [num_us]
                         (parents: [num_bones]i32)
                         (base_relatives: [num_bones][4][4]f64)
                         (inverse_base_absolutes: [num_bones][4][4]f64)
                         (weights: [num_bones][M]f64)
                         (base_positions: [4][M]f64)
                         (triangles: [][3]i32)
                         (is_mirrored: bool)
                         (correspondences: [N]i32)
                         (points: [3][N]f64)
                         (theta: [theta_count]f64)
                         (us: [num_us]f64) : [][N * 3]f64 =
  let model: hand_model [num_bones] [M] =
    { parents
    , base_relatives
    , inverse_base_absolutes
    , weights
    , base_positions
    , triangles
    , is_mirrored
    }
  let f s = jvp (uncurry (objective model correspondences points)) (theta, us) s
  let J =
    #[flattening(sequentialise_nonuniform)]
    map flatten (map (f <-< seed num_us) (iota (num_seeds num_us)))
  in reorder us J

-- | A single 'jmp' over all seeds: the primal is computed once.
entry calculate_jacobian_vec [num_bones] [N] [M] [num_us]
                             (parents: [num_bones]i32)
                             (base_relatives: [num_bones][4][4]f64)
                             (inverse_base_absolutes: [num_bones][4][4]f64)
                             (weights: [num_bones][M]f64)
                             (base_positions: [4][M]f64)
                             (triangles: [][3]i32)
                             (is_mirrored: bool)
                             (correspondences: [N]i32)
                             (points: [3][N]f64)
                             (theta: [theta_count]f64)
                             (us: [num_us]f64) : [][N * 3]f64 =
  let model: hand_model [num_bones] [M] =
    { parents
    , base_relatives
    , inverse_base_absolutes
    , weights
    , base_positions
    , triangles
    , is_mirrored
    }
  let seeds = map (seed num_us) (iota (num_seeds num_us))
  let J =
    #[flattening(sequentialise_nonuniform)]
    map flatten (jmp (uncurry (objective model correspondences points))
                     (theta, us)
                     seeds)
  in reorder us J
