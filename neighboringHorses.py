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
        
    def __set_internal_dict__(self, new_dict):
        self.neighboring_horses = new_dict
        
    # Return the neighbors of a specific horse, must be in the input set
    def get_neighbors(self, horse):
        # Return the neighbors of a horse if it's in the dict
        if horse in self.neighboring_horses.keys():
            return self.neighboring_horses[horse]
        # Otherwise return an empty set since we don't recognize the horse
        return set()
        
    def __getitem__(self, index):
        return self.get_neighbors(index)
        
        

# Construction is O(n), indexing all neighbors of every horse is O(n^2) time        
class AllNeighbors(NeighboringHorsesInterface):
    def __init__(self, input_horses):
        # Store horse neighbors in a dict
        temp_dict = dict()
        
        # Assign every horse as a neighbor
        for horse in input_horses:
            temp_dict[horse] = set(input_horses)
            # Remove the horse used as a key in neighbor list
            temp_dict[horse].remove(horse)
            
        # Update the internal dictionary
        self.__set_internal_dict__(temp_dict)
        
        

# Construction is O(nlogn) time, indexing all neighbors of every horse is O(n) time
class VoronoiNeighbors(NeighboringHorsesInterface):
    def __init__(self, input_horses):
        # Store horse neighbors in a dict
        temp_dict = dict()
        
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
            if horse_1 not in temp_dict.keys():
                temp_dict[horse_1] = set()
            if horse_2 not in temp_dict.keys():
                temp_dict[horse_2] = set()
                
            # Add neighboring horses to each set
            temp_dict[horse_1].add(horse_2)
            temp_dict[horse_2].add(horse_1)
            
        # Update the internal dictionary
        self.__set_internal_dict__(temp_dict)
        
        

# Construction is O(n^2) time, indexing all neighbors of every horse is O(n^2) time but time heavily relies on horse density and velocities.
# Likely average is O(n) for both construction and neighbors since the number of neighbors determined very rarely exceeds 9 
# (1 sprite in each cell which is the size of 1 sprite and sprites don't typically overlap)
class GridNeighbors(NeighboringHorsesInterface):
    def __init__(self, input_horses, extra_size=0):
        assert(extra_size >= 0)
        
        # Store horse neighbors in a dict
        temp_dict = dict()
        
        # Calculate each horses' center points
        horse_positions = np.array([np.array(horse.get_position()) + (np.array(horse.get_image_numpy().shape[1::-1]) / 2.0) for horse in input_horses])
        
        # Calculate grid resolution
        horse_sprite_sizes = np.array([horse.get_image_numpy().shape for horse in input_horses])
        height, width, *_ = np.max(horse_sprite_sizes, axis=0)
        
        # Create a dictionary to store each horse in cells and another to store what cell each horse is in
        grid_cells = dict()
        horse_cells = dict()
        
        # Assign each horse into a cell
        for i in range(len(horse_positions)):
            pos_x, pos_y = horse_positions[i]
            horse = input_horses[i]
            
            # Calculate the cell this horse is in
            curr_grid_cell = (int(pos_x / (width + extra_size)), int(pos_y / (height + extra_size)))
            
            # Put this horse in said cell
            if curr_grid_cell not in grid_cells.keys():
                grid_cells[curr_grid_cell] = set()
            grid_cells[curr_grid_cell].add(horse)
            
            # Remember this horse is in that cell
            horse_cells[horse] = curr_grid_cell
            
        # Now that they're in cells, let's collect all horses in the 3x3 set of cells surrounding the horse to determine the neighbors
        for horse in input_horses:
            # Create the set of neighbors for this horse
            temp_dict[horse] = set()
            
            # Determine what cell the current horse is in
            grid_pos_x, grid_pos_y = horse_cells[horse]
            # Check all surrounding cells and collect their horses
            for grid_index_x in range(grid_pos_x - 1, grid_pos_x + 2):
                for grid_index_y in range(grid_pos_y - 1, grid_pos_y + 2):
                    cell_index = (grid_index_x, grid_index_y)
                    if cell_index in grid_cells.keys():
                        for neighboring_horse in grid_cells[cell_index]:
                            temp_dict[horse].add(neighboring_horse)
            
        # Update the internal dictionary
        self.__set_internal_dict__(temp_dict)