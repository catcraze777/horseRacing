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

PHYSICS_STEPS_PER_FRAME = 50
DEBUG_OUTPUT = False
SHOW_FPS = True
FPS_FRAME_INTERVAL = 10000

ANGLE_OFFSET_INTEGRAL_STEPS = 1000  # More steps = more possible reflection offsets but is more expensive computationally
ENABLE_REFLECTION_OFFSET = True
REFLECTION_EXPONENT = 10.0      # Higher exponent = smaller random offset range

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

    # Create padding for speeding up convolution colission calculations
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
                collision_center_x = 0.0
                collision_center_y = 0.0
                collision_count = 0.0
                
                for y in range(collision_intersection.shape[0]):
                    for x in range(collision_intersection.shape[1]):
                        # If this pixel shows a collision...
                        if collision_intersection[y,x] > 0.0:
                            # Add pixel center to weighted average
                            collision_center_x += float(x) * collision_intersection[y,x]
                            collision_center_y += float(y) * collision_intersection[y,x]
                            collision_count += collision_intersection[y,x]
                            
                # Calculate weighted average of center points
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





    # Bounce a horse directly backwards with uniformly random offset within [-random_range, random_range] degrees.
    def reverse_velocity(horse, random_range):
        # Rotate by 180 degrees plus random uniform angle variation
        rotation_theta = math.pi * (180.0 + random.uniform(-random_range, random_range)) / 180.0

        # Rotate velocity vector
        new_velocity = rotate_vector(horse.get_velocity(), rotation_theta)
        
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





    # Frame length queue, used to calculate average framerate
    framelength_queue = Queue()
    framelength_sum = 0.0





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
        
        # Loop through all horses
        i = 0
        for horse in player_horses:
            # Perform velocity update PHYSICS_STEPS_PER_FRAME times
            for step in range(PHYSICS_STEPS_PER_FRAME):
                # Move horse based on velocity
                horse.velocity_tick(time_delta/1000.0/PHYSICS_STEPS_PER_FRAME)
                
                # Check if horse is hitting a wall.
                if is_colliding(horse):
                    # We collided, step backwards so we aren't colliding anymore
                    horse.velocity_tick(-2 * time_delta/1000.0/PHYSICS_STEPS_PER_FRAME)
                    
                    # If known collision location and surface normal can be calculated, reflect perfectly.
                    if collision_location != None:
                        # Calculate surface normal
                        surface_normal = calculate_normal(collision_location, 3)
                        # To avoid getting stuck in walls, don't reflect if currently moving away from wall.
                        if np.dot(horse.get_velocity(), surface_normal) < 0.0:
                            # Calculate reflected velocity
                            reflected_velocity = reflect(horse.get_velocity(), surface_normal)
                            # Apply random offset if enabled
                            offset_velocity = reflected_velocity
                            if ENABLE_REFLECTION_OFFSET:
                                offset_velocity = rotate_vector(offset_velocity, specular_reflection_random_offset(REFLECTION_EXPONENT))
                            # Apply reflected velocity
                            horse.set_velocity(offset_velocity[0], offset_velocity[1])
                        
                        # Slightly nudge velocity away from wall if a collision occurs while traveling away from wall.
                        else:
                            # Slightly nudge velocity direction away from wall
                            old_magnitude = np.linalg.norm(horse.get_velocity())
                            new_velocity = horse.get_velocity() + (surface_normal * old_magnitude / 10.0)
                            # Conserve original speed
                            new_velocity_x, new_velocity_y = (new_velocity / np.linalg.norm(new_velocity)) * old_magnitude
                            # Update velocity
                            horse.set_velocity(new_velocity_x, new_velocity_y)
                    
                    # Unknown surface normal, just go backwards.
                    else:
                        # Reverse velocity with uniform offset
                        reverse_velocity(horse, 80.0)
                    
                    #new_velocity_x = (1.2 ** random.uniform(-1,1)) * horse.get_velocity()[0]
                    #new_velocity_y = (1.2 ** random.uniform(-1,1)) * horse.get_velocity()[1]
                    
                    #horse.set_velocity(new_velocity_x, new_velocity_y)
                
                # Debug output
                debug_print(i, bool(is_colliding(horse)), np.array(horse.get_position(), dtype=int).tolist())
            
            # Update current horse number, for debugging output
            i += 1
                
            # Draw horse to screen
            horse.draw_to_surface(screen)
        
        # Frame over, this is no longer the current tick
        last_tick = current_tick

        # Display Canvas updates
        pygame.display.update()