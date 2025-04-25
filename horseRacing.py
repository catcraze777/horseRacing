# Programmed by catcraze777, for modification and free distribution with original credit and original rights.

# For easy image and sprite displays.
import pygame
from pygame.locals import *
# For file reading.
import sys
import os.path as path
# For image and matrix manipulation functions.
import cv2
import numpy as np
# For voronoi diagram class.
import scipy
# General math and rng functions.
import math
import random

# Queue FI-FO Datastructure
from queue import Queue

# Custom horse sprite object (see horse.py)
from horse import Horse

###
#
#   OPTIONS
#
###

ARENA_FILENAME = 'arena.png'
PLAYER_FILENAME_PREFIX = 'piece_'
PLAYER_FILENAME_SUFFIX = '.png'
STARTING_POSITIONS_FILENAME = 'starting_positions.txt'
STARTING_VELOCITY = [350, 0]
STARTING_VELOCITY_DIRECTION_VARIATION = 0.0 # Rotate the starting velocities by a random amount sampled uniformly within [-STARTING_VELOCITY_DIRECTION_VARIATION, STARTING_VELOCITY_DIRECTION_VARIATION] degrees

PHYSICS_STEPS_PER_FRAME = 4
DEBUG_OUTPUT = False
SHOW_FPS = False
FPS_FRAME_INTERVAL = 10000

ANGLE_OFFSET_INTEGRAL_STEPS = 1000  # More steps = more possible reflection offsets but is more expensive computationally
ENABLE_REFLECTION_OFFSET = True
REFLECTION_EXPONENT = 10.0      # Higher exponent = smaller random offset range
ENABLE_HORSE_COLLISION = True
ENABLE_HORSE_COLLISION_REFLECTION_OFFSET = False
USE_VORONOI_FOR_HORSE_COLLISIONS = False    # A voronoi diagram can speed up collision calculations by only checking neighbooring horses (O(n) checks compared to O(n^2)), but diagram construction can make this not worth it. Useful for an especially large number of horses.
BACKSTEP_SCALAR = 3.0   # Determines how far a horse travels backwards briefly if a velocity update occurs. Used to prevent horses from getting stuck upon collision
COLLISION_UPSCALING = 10     # Perform integer upscaling on the alpha channel images used for horse collisions. Simulates higher resolution sprites and positions at the cost of more expensive calculations.

USE_GOAL = True
GOAL_FILENAME = 'goal.png'
GOAL_POSITION = [137, 600]
WINNER_OUTLINE = True
WINNER_OUTLINE_THICKNESS = 3
WINNER_OUTLINE_COLOR = (1.0, 1.0, 0) # Uses decimal RGB (Multiplies alpha channel by this color)

###
#
#   CODE
#
###

# Remember, you always need to use this so if this script accidently 
# gets imported (like I imported horse.py for the Horse class) it won't run any code accidently.
if __name__ == '__main__':
    def debug_print(*args):
        if DEBUG_OUTPUT:
            print(args)

    ###
    #
    #   INITIALIZATION CODE
    #
    ###
    
    last_tick = 0.0
    
    # Ensure COLLISION_UPSCALING is formatted correctly
    COLLISION_UPSCALING = int(round(COLLISION_UPSCALING))
    if COLLISION_UPSCALING < 1:
        COLLISION_UPSCALING = 1

    # The battle arena
    arena_image_numpy = cv2.imread(ARENA_FILENAME, cv2.IMREAD_UNCHANGED)
    arena_image_pygame = pygame.image.frombuffer(arena_image_numpy[:,:,2::-1].tobytes(), arena_image_numpy.shape[1::-1], 'RGB')

    # Initialize pygame
    pygame.init()
    
    # Set window size to match size of arena image
    screen = pygame.display.set_mode((arena_image_numpy.shape[1], arena_image_numpy.shape[0]))
    
    # Set window information
    pygame.display.set_caption("HORSE RACING")
    pygame.display.set_icon(pygame.image.load("icon.png"))

    # Players
    player_images_pygame = []
    player_images_numpy = []
    player_horses = []
    player_starting_positions = None

    # Load starting positions from file.
    with open(STARTING_POSITIONS_FILENAME) as text_file:
        content = text_file.read()
        # Split by player piece
        player_starting_positions = content.split('\n')
        
        for i in range(len(player_starting_positions)):
            # Split string into list [x, y]
            player_starting_positions[i] = player_starting_positions[i].split(',')
            
            for j in range(len(player_starting_positions[i])):
                # Convert string to int
                player_starting_positions[i][j] = int(player_starting_positions[i][j].strip())

    i = 0
    # Load images for players
    curr_file_name = PLAYER_FILENAME_PREFIX + str(i) + PLAYER_FILENAME_SUFFIX
    while path.exists(curr_file_name):
        # Setup Horse sprite and images
        curr_horse = Horse(curr_file_name)
        player_images_pygame.append(curr_horse.get_image_pygame())
        player_images_numpy.append(curr_horse.get_image_numpy())
        player_horses.append(curr_horse)
        
        # Set starting conditions
        start_x, start_y = player_starting_positions[i]
        curr_horse.set_position(start_x, start_y)
        curr_horse.set_velocity(STARTING_VELOCITY[0], STARTING_VELOCITY[1])
        
        # Next file.
        i += 1
        curr_file_name = "piece_" + str(i) + ".png"
        
    image_sizes = np.array([image.shape for image in player_images_numpy])
    max_size = np.max(image_sizes, axis=0)
    
    # Goal sprite
    goal_horse = None
    goal_image_numpy = None
    goal_image_pygame = None
    
    if USE_GOAL:
        goal_horse = Horse(GOAL_FILENAME)
        goal_image_numpy = goal_horse.get_image_numpy()
        goal_image_pygame = goal_horse.get_image_pygame()
        goal_horse.set_position(GOAL_POSITION[0], GOAL_POSITION[1])
        
    # Track if a horse won or not
    winning_horse = None
    winning_horse_outline = None

    # Create padding for speeding up convolution collision calculations
    arena_image_padded = np.pad(arena_image_numpy, ((max_size[0], max_size[0]),
                                                    (max_size[1], max_size[1]),
                                                    (0, 0)))




    
    ###
    #
    #   HELPER FUNCTIONS
    #
    ###
    
    # Check if a horse's alpha channel overlaps with the arena's alpha channel.
    # collision_location is set to a List[int, int] or None to indicate center of all collision pixels
    collision_location = None
    def is_colliding(horse):
        # Make sure to modify variable outside of function.
        global collision_location
        
        # Get horse info
        pos_x, pos_y = horse.get_position()
        horse_image = horse.get_image_numpy()
        size_y, size_x, _ = horse_image.shape
        
        # Get position of sprite in the padded arena image
        pos_y_padded, pos_x_padded = [ [int(pos_y), int(pos_x)][i] + max_size[i] for i in range(2) ]
        
        # Is horse beyond image boundaries?
        if (pos_x_padded < 0 or pos_x_padded >= arena_image_padded.shape[1] or
                pos_y_padded < 0 or pos_y_padded >= arena_image_padded.shape[0]):
            # If outside boundaries, consider it a wall collision
            collision_location = None
            return True
            
        # Get alpha channel of arena section and the horse sprite
        image_section = arena_image_padded[pos_y_padded: pos_y_padded + size_y, pos_x_padded : pos_x_padded + size_x, 3]
        horse_collider = horse_image[:,:,3]
        
        try:
            # A pixel is greater than 0.0 iff that pixel is greater than 0.0 in both the horse and arena collision
            collision_intersection = image_section * horse_collider
            
            # If any of the horse's alpha pixels overlap the arena boundaries, consider it a wall collision
            is_collision = np.max(collision_intersection) > 0.0
            
            # Calculate center of all weighted collisions 
            # (Gives more influence to opaque pixels instead of transparent ones)
            # Note that although we can determine if a collision occurs here but 
            #   numpy is quicker at determining that so we arent' doing that.
            if is_collision:
                # Create array of coordinate positions
                width, height = collision_intersection.shape
                collision_center_y, collision_center_x = np.mgrid[:width, :height].astype(float)
                
                # Weight each position by the collision intensity
                collision_center_x = np.sum(collision_center_x * collision_intersection)
                collision_center_y = np.sum(collision_center_y * collision_intersection)
                collision_count = np.sum(collision_intersection)

                # Calculate weighted average of collision points
                collision_center_x = collision_center_x / collision_count
                collision_center_y = collision_center_y / collision_count 
                collision_location = [int(collision_center_x + pos_x), int(collision_center_y + pos_y)]
            else:
                # No collision, no definitive location
                collision_location = None

            # Return if a collision is detected.
            return is_collision
        except IndexError:
            # Some boundary condition error, assume it is a wall collision
            collision_location = None
            return True





    # Rotate a vector by a given amount
    def rotate_vector(input_vector, delta_theta):
        # Calculate rotation matrix
        sin_theta = math.sin(delta_theta)
        cos_theta = math.cos(delta_theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
        # Perform matrix multiplication to rotate vector
        rotated_vector = np.transpose(rotation_matrix @ np.transpose(np.array([input_vector])))[0]
        
        return rotated_vector





    # Reverse the direction of a velocity with uniformly random offset within [-random_range, random_range] degrees.
    def reverse_velocity(velocity, random_range):
        new_velocity = -1 * np.array(velocity)
        if abs(random_range) > 0:
            # Rotate by 180 degrees plus random uniform angle variation
            rotation_theta = math.pi * (random.uniform(-random_range, random_range)) / 180.0

            # Rotate velocity vector
            new_velocity = rotate_vector(new_velocity, rotation_theta)
            
        return new_velocity
    
    # Bounce a horse directly backwards with uniformly random offset within [-random_range, random_range] degrees.
    def reverse_horse_velocity(horse, random_range):
        # Create new velocity vector
        new_velocity = reverse_velocity(horse.get_velocity(), random_range)
        
        # Apply rotated velocity vector
        horse.set_velocity(new_velocity[0], new_velocity[1])





    # Calculate estimated normal vector given a center coordinate and range.
    # Samples arena area from [-range + center[0]: center[0] + range, -range + center[1]: center[1] + range]
    def calculate_normal(center, range_offset):
        padding_y, padding_x = max_size[:2]
        center_x, center_y = center
        
        # Get relevant section of the arena
        image_section = arena_image_padded[padding_y + center_y - range_offset: padding_y + center_y + range_offset + 1,
                                            padding_x + center_x - range_offset: padding_x + center_x + range_offset + 1,
                                            3]
        
        # Find gradient values that estimate the derivative on each axis
        gradient_x = cv2.Sobel(image_section, cv2.CV_16S, 1, 0)[1:-1, 1:-1]
        gradient_y = cv2.Sobel(image_section, cv2.CV_16S, 0, 1)[1:-1, 1:-1]
        
        # Change gradient weights based on how close they are to the center
        gaussian_kernel_1D = cv2.getGaussianKernel(gradient_x.shape[0], range_offset / 3.0)
        gaussian_kernel_2D = gaussian_kernel_1D @ np.transpose(gaussian_kernel_1D)
        gradient_x = gradient_x * gaussian_kernel_2D
        gradient_y = gradient_y * gaussian_kernel_2D
        
        # Sum the gradients
        average_gradient_x = np.sum(gradient_x)
        average_gradient_y = np.sum(gradient_y)
        
        
        # Create and normalize the normal vector
        normal_vector = np.array([average_gradient_x, average_gradient_y])
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        
        # Correct normal vector to face away from the wall edge
        return -1 * normal_vector





    # Perfectly reflect an input vector based on an input surface normal.
    def reflect(input_vector, normal):
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Calculate input vector projected onto the normal vector
        input_projected_onto_normal = np.dot(input_vector, normal) * normal
        
        # Return reflected vector
        return -2 * input_projected_onto_normal + input_vector





    # Calculate base distribution PDF for specular reflection angle offset
    specular_reflection_pdf = np.zeros((ANGLE_OFFSET_INTEGRAL_STEPS,))
    
    starting_theta = - math.pi / 2.0
    current_theta = starting_theta
    theta_step = math.pi / ANGLE_OFFSET_INTEGRAL_STEPS
    for step in range(ANGLE_OFFSET_INTEGRAL_STEPS):
        # Calculate "probability" of reflecting to this angle
        specular_reflection_pdf[step] = math.cos(current_theta)
        
        # Advance theta
        current_theta += theta_step
        
    # Store any CDFs created during specular_reflection_random_offset() calls (Memoization kinda)
    calculated_cdfs = dict()
        
    # Specular reflection random theta offset distribution. Based on phong specular reflectance model.
    def specular_reflection_random_offset(shine_exponent=1.0):
        exponent_specular_reflection_cdf = None
        
        # Did we calculate the CDF of this shine exponent?
        if shine_exponent in calculated_cdfs.keys():
            # If yes load the CDF already calculated
            exponent_specular_reflection_cdf = calculated_cdfs[shine_exponent]
        else:
            # If no calculate the CDF and save it for later
            # Apply specular exponent
            exponent_specular_reflection_pdf = specular_reflection_pdf ** shine_exponent
            
            # Given the above pdf, calculate it's cdf
            exponent_specular_reflection_cdf = np.zeros((ANGLE_OFFSET_INTEGRAL_STEPS,))
            exponent_specular_reflection_cdf[0] = exponent_specular_reflection_pdf[0]
            for i in range(1,ANGLE_OFFSET_INTEGRAL_STEPS):
                exponent_specular_reflection_cdf[i] = exponent_specular_reflection_cdf[i - 1] + exponent_specular_reflection_pdf[i]
                
            # Normalize the cdf to total probability of 1
            exponent_specular_reflection_cdf = exponent_specular_reflection_cdf / exponent_specular_reflection_cdf[-1]
            
            # Save CDF
            calculated_cdfs[shine_exponent] = exponent_specular_reflection_cdf
            
        assert(exponent_specular_reflection_cdf is not None)
        
        # Draw from distribution
        random_number = random.random()
        
        # Find index of cdf that is closest to random_number
        cdf_difference = exponent_specular_reflection_cdf - random_number
        cdf_difference_abs = np.abs(cdf_difference)
        closest_index = np.argmin(cdf_difference_abs)
        
        # Return theta offset
        return starting_theta + closest_index * theta_step





    # Calculate overlap of two bounding boxes specified by two corners. Overlap returned is relative to first bounding box.
    def bounding_box_overlap(box1, box2):
        x_overlap = [0, box1[1][0] - box1[0][0]]
        if box2[1][0] < box1[1][0]:
            x_overlap[1] = x_overlap[1] - (box1[1][0] - box2[1][0])
        if box2[0][0] > box1[0][0]:
            x_overlap[0] = x_overlap[0] + (box2[0][0] - box1[0][0])
            
        y_overlap = [0, box1[1][1] - box1[0][1]]
        if box2[1][1] < box1[1][1]:
            y_overlap[1] = y_overlap[1] - (box1[1][1] - box2[1][1])
        if box2[0][1] > box1[0][1]:
            y_overlap[0] = y_overlap[0] + (box2[0][1] - box1[0][1])
            
        return x_overlap, y_overlap
        
    # Calculate if two horses overlap
    def do_horses_collide(horse1, horse2):
        # Calculate bounding boxes
        horse1_position = (np.array(horse1.get_position()) * COLLISION_UPSCALING).astype(int)
        horse1_size = np.array(horse1.get_image_numpy().shape[:2]) * COLLISION_UPSCALING
        horse1_bounding_corners = [horse1_position, horse1_position + horse1_size]
        
        horse2_position = (np.array(horse2.get_position()) * COLLISION_UPSCALING).astype(int)
        horse2_size = np.array(horse2.get_image_numpy().shape[:2]) * COLLISION_UPSCALING
        horse2_bounding_corners = [horse2_position, horse2_position + horse2_size]
        
        # Calculate overlapping section
        
        # Calculate overlap with respect to horse1's sprite
        x_overlap_1, y_overlap_1 = bounding_box_overlap(horse1_bounding_corners, horse2_bounding_corners)
            
        # Check if x_overlap_1 doesn't exist
        if x_overlap_1[0] >= x_overlap_1[1]:
            return False
            
        # Check if y_overlap_1 doesn't exist
        if y_overlap_1[0] >= y_overlap_1[1]:
            return False
            
        # If we haven't returned False yet, then an overlap might exist
        
        # Calculate overlap with respect to horse2's sprite. This must exist if the previous overlap did
        x_overlap_2, y_overlap_2 = bounding_box_overlap(horse2_bounding_corners, horse1_bounding_corners)
        
        # Get each horse's alpha channel
        horse1_overlap = horse1.get_image_numpy()[:,:, 3]
        horse2_overlap = horse2.get_image_numpy()[:,:, 3]
        
        # Integer upscale to simulate higher precision
        horse1_overlap = cv2.resize(horse1_overlap, horse1_size, interpolation=cv2.INTER_NEAREST)
        horse2_overlap = cv2.resize(horse2_overlap, horse2_size, interpolation=cv2.INTER_NEAREST)
        
        # Get the overlapping section
        horse1_overlap = horse1_overlap[x_overlap_1[0]:x_overlap_1[1], y_overlap_1[0]:y_overlap_1[1]]
        horse2_overlap = horse2_overlap[x_overlap_2[0]:x_overlap_2[1], y_overlap_2[0]:y_overlap_2[1]]
        
        # If sizes differ due to collision upscaling, try to rectify it and set them to the same overlapping area
        if horse1_overlap.shape != horse2_overlap.shape:
            min_size = np.min(np.array([horse1_overlap.shape, horse2_overlap.shape]), axis=0)
            horse1_overlap = horse1_overlap[:min_size[0], :min_size[1]]
            horse2_overlap = horse2_overlap[:min_size[0], :min_size[1]]
        
        # Like arena collision, these horses overlap if both alpha pixels are > 0.0, so their product must be > 0.0
        multiply_overlap = horse1_overlap * horse2_overlap
        
        # Return that a collision occured if any overlap pixel > 0.0 and ensure non-zero shape
        return np.min(multiply_overlap.shape) > 0 and np.max(multiply_overlap) > 0.0
        
        
    
    
    
    # Frame length queue, used to calculate average framerate
    framelength_queue = Queue()
    framelength_sum = 0.0
    
    # Randomly vary horse starting velocities.
    STARTING_VELOCITY_DIRECTION_VARIATION = abs(STARTING_VELOCITY_DIRECTION_VARIATION)
    if STARTING_VELOCITY_DIRECTION_VARIATION > 0.0:
        for horse in player_horses:
            rotation_theta = math.pi * (random.uniform(-STARTING_VELOCITY_DIRECTION_VARIATION, STARTING_VELOCITY_DIRECTION_VARIATION)) / 180.0
            new_velocity_x, new_velocity_y = rotate_vector(horse.get_velocity(), rotation_theta)
            horse.set_velocity(new_velocity_x, new_velocity_y)





    ###
    #
    #   GAME LOOP
    #
    ###
    while True:
        # If window closed exit program
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        
        # Calculate elapsed time
        current_tick = pygame.time.get_ticks()
        time_delta = current_tick - last_tick
        
        if SHOW_FPS:
            # Calculate sum of all frame lengths in the queue
            framelength_queue.put(time_delta)
            framelength_sum += time_delta
            
            # If queue too long, dequeue a frame length and update sum.
            if framelength_queue.qsize() >= FPS_FRAME_INTERVAL:
                removed_framelength = framelength_queue.get()
                framelength_sum -= removed_framelength
                
            # Calculate average frame length (in milliseconds)
            average_framelength = framelength_sum / framelength_queue.qsize()
            
            # If non-zero average_framelength, display FPS
            if average_framelength > 0.0:
                print("FPS:", 1.0 / (average_framelength / 1000.0))
        
        # Draw arena
        screen.blit(arena_image_pygame, (0,0))
        
        # Perform velocity update PHYSICS_STEPS_PER_FRAME times
        for step in range(PHYSICS_STEPS_PER_FRAME):
            # Determine neighbooring horses for horse collision calculations
            neighbooring_horses = None
            if ENABLE_HORSE_COLLISION:
                neighbooring_horses = dict()
                horse_positions = np.array([np.array(horse.get_position()) + (np.array(horse.get_image_numpy().shape[:2]) / 2.0) for horse in player_horses])
                # Are we using a voronoi diagram to only check neighbooring horses?
                if USE_VORONOI_FOR_HORSE_COLLISIONS:
                    # Create voronoi diagram to find neighboors
                    horse_voronoi = None
                    while horse_voronoi is None:
                        try:
                            horse_voronoi = scipy.spatial.Voronoi(horse_positions)
                        except scipy.spatial._qhull.QhullError:
                            # Inputs have an edge case of same coordinate value, add slight offset to points and try again.
                            for i in range(len(horse_positions)):
                                for j in range(len(horse_positions[i])):
                                    horse_positions[i][j] += random.random() * 0.001
                    
                    # Use ridge_points to find neighbooring horses.
                    for horse_1, horse_2 in horse_voronoi.ridge_points:
                        # ridge_points returns point indices, turn them into horse objects.
                        horse_1 = player_horses[horse_1]
                        horse_2 = player_horses[horse_2]
                        # Create empty sets if horse not in dict
                        if horse_1 not in neighbooring_horses.keys():
                            neighbooring_horses[horse_1] = set()
                        if horse_2 not in neighbooring_horses.keys():
                            neighbooring_horses[horse_2] = set()
                            
                        # Add neighbooring horses to each set
                        neighbooring_horses[horse_1].add(horse_2)
                        neighbooring_horses[horse_2].add(horse_1)
                        
                # Use brute force approach and check every other horse for collisions. 
                else:
                    for horse in player_horses:
                        neighbooring_horses[horse] = player_horses
                
            # Store horses whose velocities need to be updated
            update_horses = dict()
            
            # Update the dict with new velocities
            def update_horse_velocity(horse, new_velocity):
                if horse in update_horses.keys():
                    update_horses[horse] += new_velocity
                else:
                    update_horses[horse] = new_velocity
            
            # Loop through all horses
            i = 0
            for horse in player_horses:
                # Check if horse is hitting a wall.
                if is_colliding(horse):
                    # If known collision location and surface normal can be calculated, reflect perfectly.
                    if collision_location != None:
                        # Calculate surface normal
                        surface_normal = calculate_normal(collision_location, 3)
                        
                        # If stuck, try to move directly away from the wall.
                        if horse.is_stuck():
                            new_velocity = surface_normal / np.linalg.norm(surface_normal) * np.linalg.norm(horse.get_velocity())
                            update_horse_velocity(horse, new_velocity)
                            continue
                            
                        # To avoid getting stuck in walls, don't reflect if currently moving away from wall.
                        if np.dot(horse.get_velocity(), surface_normal) < 0.0:
                            # Calculate reflected velocity
                            reflected_velocity = reflect(horse.get_velocity(), surface_normal)
                            # Apply random offset if enabled
                            offset_velocity = reflected_velocity
                            if ENABLE_REFLECTION_OFFSET:
                                offset_velocity = rotate_vector(offset_velocity, specular_reflection_random_offset(REFLECTION_EXPONENT))
                            # Save reflected velocity
                            update_horse_velocity(horse, offset_velocity)
                        
                        # Slightly nudge velocity away from wall if a collision occurs while traveling away from wall.
                        else:
                            # Slightly nudge velocity direction away from wall
                            old_magnitude = np.linalg.norm(horse.get_velocity())
                            new_velocity = horse.get_velocity() + (surface_normal * old_magnitude / 10.0)
                            # Conserve original speed
                            new_velocity = (new_velocity / np.linalg.norm(new_velocity)) * old_magnitude
                            # Update velocity
                            update_horse_velocity(horse, new_velocity)
                    
                    # Unknown surface normal, just go backwards.
                    else:
                        # Reverse velocity with uniform offset
                        new_velocity = reverse_velocity(horse.get_velocity(), 80.0)
                        # Update velocity
                        update_horse_velocity(horse, new_velocity)
                    
                    #new_velocity_x = (1.2 ** random.uniform(-1,1)) * horse.get_velocity()[0]
                    #new_velocity_y = (1.2 ** random.uniform(-1,1)) * horse.get_velocity()[1]
                    
                    #horse.set_velocity(new_velocity_x, new_velocity_y)
                
                # Check horse collisions.
                if neighbooring_horses is not None:
                    for other_horse in neighbooring_horses[horse]:
                        # The same horse will collide with itself.
                        if horse == other_horse:
                            continue
                        # Check if two horses collide.
                        if do_horses_collide(horse, other_horse):
                            # Use relative positioning to estimate a reflection normal.
                            relative_position_vector = np.array(horse.get_position()) - np.array(other_horse.get_position())
                            relative_position_vector += (np.array(horse.get_image_numpy().shape[:2]) - np.array(other_horse.get_image_numpy().shape[:2])) / 2.0
                            relative_position_vector = relative_position_vector / np.linalg.norm(relative_position_vector)
                            
                            # If a horse thinks it's stuck, have them go directly away from each other
                            if horse.is_stuck() or other_horse.is_stuck():
                                new_velocity_1 = relative_position_vector / np.linalg.norm(relative_position_vector) * np.linalg.norm(horse.get_velocity())
                                new_velocity_2 = -1 * relative_position_vector / np.linalg.norm(relative_position_vector) * np.linalg.norm(other_horse.get_velocity())
                                update_horse_velocity(horse, new_velocity_1)
                                update_horse_velocity(other_horse, new_velocity_2)
                                continue
                            
                            # Test if horses are approaching eachother
                            projected_velocity_1 = np.dot(horse.get_velocity(), relative_position_vector) * relative_position_vector
                            projected_velocity_2 = np.dot(other_horse.get_velocity(), relative_position_vector) * relative_position_vector
                            
                            # They aren't approaching eachother if they aren't going in the same direction and other_horse is not approaching horse (different direction as relative_position_vector)
                            if np.dot(projected_velocity_1, projected_velocity_2) < 0.0 and np.dot(projected_velocity_2, relative_position_vector) < 0.0:
                                # If they are moving away, don't do any velocity updates
                                continue
                            
                            # Calculate reflected velocity.
                            reflected_velocity_1 = reflect(horse.get_velocity(), relative_position_vector)
                            reflected_velocity_2 = reflect(other_horse.get_velocity(), -1 * relative_position_vector)
                            
                            # Apply random offset if enabled
                            offset_velocity_1 = reflected_velocity_1
                            offset_velocity_2 = reflected_velocity_2
                            if ENABLE_HORSE_COLLISION_REFLECTION_OFFSET:
                                offset_velocity_1 = rotate_vector(offset_velocity_1, specular_reflection_random_offset(REFLECTION_EXPONENT))
                                offset_velocity_2 = rotate_vector(offset_velocity_2, specular_reflection_random_offset(REFLECTION_EXPONENT))
                                
                            # Test if horses are now approaching each other
                            projected_velocity_1 = np.dot(offset_velocity_1, relative_position_vector) * relative_position_vector
                            projected_velocity_2 = np.dot(offset_velocity_2, relative_position_vector) * relative_position_vector
                            
                            # They are approaching eachother if they aren't going in the same direction and other_horse is approaching horse (same direction as relative_position_vector)
                            if np.dot(projected_velocity_1, projected_velocity_2) < 0.0 and np.dot(projected_velocity_2, relative_position_vector) > 0.0:
                                # Stop them from approaching eachother with slight push away from eachother.
                                offset_velocity_1 = offset_velocity_1 - 1.1 * projected_velocity_1
                                offset_velocity_2 = offset_velocity_2 - 1.1 * projected_velocity_2
                            
                            # Save reflected velocity
                            update_horse_velocity(horse, offset_velocity_1)
                            update_horse_velocity(other_horse, offset_velocity_2)
                        
                        
                # Debug output
                debug_print(i, bool(is_colliding(horse)), np.array(horse.get_position(), dtype=int).tolist())
            
                # Update current horse number, for debugging output
                i += 1
                
            # Perform a physics tick update.
            for horse in player_horses:
                # Update velocities of all horses that collided with something.
                if horse in update_horses.keys():
                    # New velocity is equal to the sum of velocity updates normalized to match the original speed before the update.
                    new_velocity = update_horses[horse] / np.linalg.norm(update_horses[horse]) * np.linalg.norm(horse.get_velocity())
                    
                    # We collided with something, step backwards so we aren't colliding anymore
                    if BACKSTEP_SCALAR > 0.0:
                        original_position = (np.array(horse.get_position()) / BACKSTEP_SCALAR).astype(int)
                        while np.all((np.array(horse.get_position()) / BACKSTEP_SCALAR).astype(int) == original_position):
                            horse.velocity_tick(-1 * time_delta/1000.0/PHYSICS_STEPS_PER_FRAME)
                    
                    horse.set_velocity(new_velocity[0], new_velocity[1])
                    
                # Update positions.
                horse.velocity_tick(time_delta/1000.0/PHYSICS_STEPS_PER_FRAME)
        
        # Check goal collisions to determine a winner.
        if USE_GOAL and winning_horse is None:
            goal_horse.draw_to_surface(screen)
            
            # Check to see if any horse is touching the goal.
            for horse in player_horses:
                # Does this horse collide with the goal?
                if do_horses_collide(horse, goal_horse):
                    # Yes! This horse is now the winner.
                    winning_horse = horse
                    
                    # Begin creating the winner's outline.
                    winning_horse_outline_alpha_channel = np.pad(horse.get_image_numpy()[:,:,3], WINNER_OUTLINE_THICKNESS)
                    
                    # Perform an image dilation.
                    outline_filter = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*WINNER_OUTLINE_THICKNESS, 2*WINNER_OUTLINE_THICKNESS))
                    winning_horse_outline_alpha_channel = cv2.dilate(winning_horse_outline_alpha_channel, outline_filter)
                    
                    # Create the image
                    height, width = winning_horse_outline_alpha_channel.shape
                    winning_horse_outline_img = np.full((height, width, 4), 255, dtype=np.uint8)
                    # Set outline color
                    winning_horse_outline_img[:,:,:3] = winning_horse_outline_img[:,:,:3] * WINNER_OUTLINE_COLOR
                    # Set outline transparency
                    winning_horse_outline_img[:,:,3] = winning_horse_outline_img[:,:,3] * (winning_horse_outline_alpha_channel / np.max(winning_horse_outline_alpha_channel))
                    
                    # Convert the numpy image into a pygame Canvas to draw to screen
                    winning_horse_outline = pygame.image.frombuffer(winning_horse_outline_img.tobytes(), winning_horse_outline_img.shape[1::-1], 'RGBA')
                    
        # Else if a winning horse exists, draw it's stored outline
        elif winning_horse is not None and winning_horse_outline is not None:
            outline_position = np.array(winning_horse.get_position()) - np.array((WINNER_OUTLINE_THICKNESS, WINNER_OUTLINE_THICKNESS))
            screen.blit(winning_horse_outline, tuple(outline_position))
        
        # Draw all horses to the screen.
        for horse in player_horses:
            horse.draw_to_surface(screen)
        
        # Frame over, this is no longer the current tick
        last_tick = current_tick

        # Display Canvas updates
        pygame.display.update()