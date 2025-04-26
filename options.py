###
#
#   OPTIONS
#
###

ARENA_FILENAME = 'arena.png'
PLAYER_FILENAME_PREFIX = 'piece_'
PLAYER_FILENAME_SUFFIX = '.png'
STARTING_POSITIONS_FILENAME = 'starting_positions.txt'
STARTING_VELOCITY = [150, 0]
STARTING_VELOCITY_DIRECTION_VARIATION = 0.0 # Rotate the starting velocities by a random amount sampled uniformly within [-STARTING_VELOCITY_DIRECTION_VARIATION, STARTING_VELOCITY_DIRECTION_VARIATION] degrees

PHYSICS_STEPS_PER_FRAME = 4
DEBUG_OUTPUT = False
SHOW_FPS = True
FPS_FRAME_INTERVAL = 1000

ANGLE_OFFSET_INTEGRAL_STEPS = 1000  # More steps = more possible reflection offsets but is more expensive computationally
ENABLE_REFLECTION_OFFSET = True
REFLECTION_EXPONENT = 10.0      # Higher exponent = smaller random offset range
ENABLE_HORSE_COLLISION = True
ENABLE_HORSE_COLLISION_REFLECTION_OFFSET = False
HORSE_DIVERGENCE_REQUIREMENT = 10.0     # If two horses collide and their velocities are in roughly the same direction within a degree tolerance specified here, apply a small force to push them a part
HORSE_DIVERGENCE_FACTOR = 0.3       # If two horses meet the above requirement, the force applied scales with this factor
USE_VORONOI_FOR_HORSE_COLLISIONS = True    # A voronoi diagram can speed up collision calculations by only checking neighbooring horses (O(n) checks compared to O(n^2)), but diagram construction can make this not worth it. Useful for an especially large number of horses.
BACKSTEP_SCALAR = 0.0   # Determines how far a horse travels backwards briefly if a velocity update occurs. Used to prevent horses from getting stuck upon collision
COLLISION_UPSCALING = 10     # Perform integer upscaling on the alpha channel images used for horse collisions. Simulates higher resolution sprites and positions at the cost of more expensive calculations.

USE_GOAL = False
GOAL_FILENAME = 'goal.png'
GOAL_POSITION = [137, 600]
WINNER_OUTLINE = True
WINNER_OUTLINE_THICKNESS = 3
WINNER_OUTLINE_COLOR = (1.0, 1.0, 0) # Uses decimal RGB (Multiplies alpha channel by this color)