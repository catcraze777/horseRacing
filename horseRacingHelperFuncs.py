# For image and matrix manipulation functions.
import cv2
import numpy as np
# General math and rng functions.
import math
import random

# Custom horse sprite object (see horse.py)
from horse import Horse

# Import option constants
from options import *

###
#
#   HELPER FUNCTIONS
#
###



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
    
    # Downscale calculated overlaps
    x_overlap_1 = [int(x_overlap_1[0] / float(COLLISION_UPSCALING)), int(math.ceil(x_overlap_1[1] / float(COLLISION_UPSCALING)))]
    x_overlap_2 = [int(x_overlap_2[0] / float(COLLISION_UPSCALING)), int(math.ceil(x_overlap_2[1] / float(COLLISION_UPSCALING)))]
    y_overlap_1 = [int(y_overlap_1[0] / float(COLLISION_UPSCALING)), int(math.ceil(y_overlap_1[1] / float(COLLISION_UPSCALING)))]
    y_overlap_2 = [int(y_overlap_2[0] / float(COLLISION_UPSCALING)), int(math.ceil(y_overlap_2[1] / float(COLLISION_UPSCALING)))]
    
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
    return np.min(multiply_overlap.shape) > 0 and np.any(multiply_overlap > 0.0)





# Normalize an input vector
def normalize_vector(input_vector):
    return input_vector / np.linalg.norm(input_vector)