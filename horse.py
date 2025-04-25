# Pygame image to Canvas loading.
import pygame
# Image Reading
import cv2
# Keep track of time
import time

UPDATE_BUFFER_SIZE = 100
STUCK_FREQUENCY_REQUIREMENT = 0.01
MINIMUM_UPDATES = 20
EPSILON_FACTOR = 0.1    # Some float between [0.0,1.0]. Used to make older update differences less important in determining if a horse is stuck. 1.0 is the past is equally important as the present. 0.0 is only consider right now! (Probably don't set it to 0.0)

# Object to represent "Horse" sprites and their corresponding data.
# Horses store their numpy and pygame image files and information about their position and velocities.
# Additional functions exist to modify position and velocities as well as update position over time.
# Horses also contain a function to render themselves on any input pygame Canvas.
class Horse:
    def __init__(self, image_name):
        self.horse_name = image_name
        self.image_numpy = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
        self.image_pygame = pygame.image.load(image_name)
        
        self.position = [0.0,0.0]
        self.velocity = [0.0,0.0]
        
        self.update_buffer = [None for i in range(UPDATE_BUFFER_SIZE)]
        self.update_buffer_index = 0
        
        
    # Set position of sprite.
    def set_position(self, x, y):
        self.position = [float(x), float(y)]
    
    # Get position of sprite.
    def get_position(self):
        return self.position
    
    # Translate sprite by specified delta per axis.
    def translate(self, delta_x, delta_y):
        curr_x, curr_y = self.get_position()
        self.set_position(curr_x + float(delta_x), curr_y + float(delta_y))
        
    # Store time of update.
    def _update_buffer(self):
        self.update_buffer[self.update_buffer_index] = time.time()
        self.update_buffer_index = (self.update_buffer_index + 1) % UPDATE_BUFFER_SIZE
        
    # Set the sprite velocity.
    def set_velocity(self, x, y):
        self.velocity = [float(x),float(y)]
        self._update_buffer()
        
    # Get the sprite velocity.
    def get_velocity(self):
        return self.velocity
        
    # Advance position based on current velocity and time that past.
    def velocity_tick(self, delta_t):
        delta_x, delta_y = self.get_velocity()
        self.translate(delta_x * delta_t, delta_y * delta_t)
        
    # Return the image of the sprite as a numpy matrix.
    def get_image_numpy(self):
        return self.image_numpy
        
    # Return the image of the sprite as a pygame Surface.
    def get_image_pygame(self):
        return self.image_pygame
        
    # Draw the sprite on a pygame Surface.
    def draw_to_surface(self, target_surface):
        target_surface.blit(self.get_image_pygame(), tuple(self.get_position()))
        
    # Attempt to determine if the sprite is stuck based on past velocity updates.
    def is_stuck(self):
        curr_index_1 = (self.update_buffer_index - 1) % UPDATE_BUFFER_SIZE
        curr_index_2 = (curr_index_1 - 1) % UPDATE_BUFFER_SIZE
        update_count = 0
        average_update_time_difference = 0.0
        
        current_time = time.time()
        
        # Find time difference between all velocity updates
        while (curr_index_1 != self.update_buffer_index and curr_index_2 != self.update_buffer_index and 
                self.update_buffer[curr_index_1] is not None and self.update_buffer[curr_index_2] is not None):
            # Find difference between these two updates
            average_update_time_difference += self.update_buffer[curr_index_1] - self.update_buffer[curr_index_2]
            
            # Update indices
            curr_index_1 = curr_index_2
            curr_index_2 = (curr_index_2 - 1) % UPDATE_BUFFER_SIZE
            
            # Add one to update differences found
            update_count += EPSILON_FACTOR ** (current_time - self.update_buffer[curr_index_1])
        
        # If too few updates then assume we're not stuck.
        if update_count < MINIMUM_UPDATES:
            return False
            
        average_update_time_difference = average_update_time_difference / float(update_count)
        
        am_i_stuck = average_update_time_difference < STUCK_FREQUENCY_REQUIREMENT
        if am_i_stuck:
            print(self.horse_name, "STUCK!!!!")
        return am_i_stuck