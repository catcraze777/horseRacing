# Pygame image to Canvas loading.
import pygame
# Image Reading
import cv2

# Object to represent "Horse" sprites and their corresponding data.
# Horses store their numpy and pygame image files and information about their position and velocities.
# Additional functions exist to modify position and velocities as well as update position over time.
# Horses also contain a function to render themselves on any input pygame Canvas.
class Horse:
    def __init__(self, image_name):
        self.image_numpy = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
        self.image_pygame = pygame.image.load(image_name)
        
        self.position = [0.0,0.0]
        self.velocity = [0.0,0.0]
        
    # Set position of sprite.
    def set_position(self, x, y):
        self.position = [x, y]
    
    # Get position of sprite.
    def get_position(self):
        return self.position
    
    # Translate sprite by specified delta per axis.
    def translate(self, delta_x, delta_y):
        curr_x, curr_y = self.get_position()
        self.set_position(curr_x + delta_x, curr_y + delta_y)
        
    # Set the sprite velocity.
    def set_velocity(self, x, y):
        self.velocity = [x,y]
        
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