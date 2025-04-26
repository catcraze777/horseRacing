# Custom horse sprite object (see horse.py)
from horse import Horse

# For voronoi diagram class.
import scipy
# For matrix manipulation functions.
import numpy as np



###
#
#   Datastructures to efficiently determine neighboring horses for collision checks.
#
###



# Create a common interface to calculate and return neighboring horses
class NeighboringHorsesInterface:
    def __init__(self, input_horses):
        pass
        
    # Return the neighbors of a specific horse, must be in the input set
    def get_neighbors(self, horse):
        pass
        
    def __getitem__(self, index):
        return self.get_neighbors(index)
        
        
        
class AllNeighbors(NeighboringHorsesInterface):
    def __init__(self, input_horses):
        # Store horse neighbors in a dict
        self.neighboring_horses = dict()
        
        # Assign every horse as a neighbor
        for horse in input_horses:
            self.neighboring_horses[horse] = set(input_horses)
            # Remove the horse used as a key in neighbor list
            self.neighboring_horses[horse].remove(horse)
            
    def get_neighbors(self, horse):
        # Return the neighbors of a horse if it's in the dict
        if horse in self.neighboring_horses.keys():
            return self.neighboring_horses[horse]
        # Otherwise return an empty set since we don't recognize the horse
        return set()
        
        
        
class VoronoiNeighbors(NeighboringHorsesInterface):
    def __init__(self, input_horses):
        # Store horse neighbors in a dict
        self.neighboring_horses = dict()
        
        # Calculate each horses' center points
        horse_positions = np.array([np.array(horse.get_position()) + (np.array(horse.get_image_numpy().shape[:2]) / 2.0) for horse in input_horses])
        
        # Create voronoi diagram to find neighbors
        horse_voronoi = None
        while horse_voronoi is None:
            try:
                horse_voronoi = scipy.spatial.Voronoi(horse_positions)
            except scipy.spatial._qhull.QhullError:
                # Inputs have an edge case of same coordinate value, add slight offset to points and try again.
                for i in range(len(horse_positions)):
                    for j in range(len(horse_positions[i])):
                        horse_positions[i][j] += random.random() * 0.001
        
        # Use ridge_points to find neighboring horses.
        for horse_1, horse_2 in horse_voronoi.ridge_points:
            # ridge_points returns point indices, turn them into horse objects.
            horse_1 = input_horses[horse_1]
            horse_2 = input_horses[horse_2]
            # Create empty sets if horse not in dict
            if horse_1 not in self.neighboring_horses.keys():
                self.neighboring_horses[horse_1] = set()
            if horse_2 not in self.neighboring_horses.keys():
                self.neighboring_horses[horse_2] = set()
                
            # Add neighboring horses to each set
            self.neighboring_horses[horse_1].add(horse_2)
            self.neighboring_horses[horse_2].add(horse_1)
            
    def get_neighbors(self, horse):
        # Return the neighbors of a horse if it's in the dict
        if horse in self.neighboring_horses.keys():
            return self.neighboring_horses[horse]
        # Otherwise return an empty list since we don't recognize the horse
        return set()