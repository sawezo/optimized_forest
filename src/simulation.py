from tqdm.notebook import tqdm
import random

import numpy as np

from scipy.ndimage.measurements import label as label_components



def create_grid(N, spark=False, spark_distribution_FUNC=None):
    
    if spark: # if a spark probability grid is desired
        spark_grid = create_grid(N)
        empty_idxs = np.ndindex(N, N)

        # filling cells of an empty grid with values from the specified spark distribution 
        for idx in empty_idxs:
            spark_grid[idx[0], idx[1]] = spark_distribution_FUNC(idx)

        # normalizing probabilities (sum to one)
        spark_grid /= np.sum(spark_grid) 
        
        return spark_grid
    
        
    else: # otherwise creating a regular empty grid
        return np.zeros((N, N))


def plant_tree(forest_grid, idx):
    
    forest_grid_test = forest_grid.copy() # avoid overwriting actual forest in memory
    forest_grid_test[idx[0], idx[1]] = 1 # planting a new tree
    
    # connected components of trees and their sizes
    components_grid, _ = label_components(forest_grid_test)
    components, counts = np.unique(components_grid, return_counts=True)
    component2size = dict(zip(components, counts))

    return forest_grid_test, component2size, components_grid


def burn_forest(forest_grid, spark_grid, connectivity, component2size, components_grid):
    
    # the expected total fire size with the newly planted tree
    expected_cost = 0 
    for idx in (np.argwhere(forest_grid==1)).tolist(): # for each tree
        tree_coords = idx[0], idx[1]
        
        # the probability of a lightning strike hitting this tree
        spark_probability = spark_grid[tree_coords]
        
        # getting the connected component and size that corresponds to this tree
        component_id = components_grid[tree_coords]

        # the size of the fire if this tree did catch (weighting the component size by probability)
        potential_fire_size = component2size[component_id] * spark_probability
        expected_cost += potential_fire_size
        

    return expected_cost


def run(L, D, connectivity, spark_distribution_FUNC):
    """
    Runs a single simulation with specified parameters.
    """
    forest_grid = create_grid(L) # creating the forest grid

    # creating the spark grid, where each value represents the probability of a spark
    spark_grid = create_grid(L, spark=True, spark_distribution_FUNC=spark_distribution_FUNC)
    
    
    # run items and structures to populate with data
    tree_ct = 0 # to tally
    empty_idxs = np.argwhere(forest_grid==0).tolist()
    
    epoch2forest_grid = dict()
    epoch2cost = dict()
    epoch2yield = dict()
    
    
    # running the simulation
    completion = tqdm(leave=False, total=len(empty_idxs), desc="Simulation (D={}, L={})".format(str(D), str(L)))
    while len(empty_idxs) > 0: # while there are still spaces left to plant a tree

        # pick 'D' cells to plant a tree
        if D <= len(empty_idxs):
            new_tree_idxs = random.sample(empty_idxs, D) 
        else: # D > number cells remaining
            new_tree_idxs = empty_idxs


        # figuring out what tree will result in the smallest expected fire burn
        new_tree2expected_cost = dict() 
        for idx in new_tree_idxs: 
            
            # plant the tree on a copy of the forest
            forest_grid_test, component2size, components_grid = plant_tree(forest_grid, idx) 
            
            # burn the forest down to assess the expected fire size and save the result
            new_tree2expected_cost[idx[0], idx[1]] = burn_forest(forest_grid_test, spark_grid, connectivity,
                                                                 component2size, components_grid)

        optimal_tree__cost = min(new_tree2expected_cost.items(), key=lambda x: x[1]) # tree with minimum cost

        
        # planting this tree in the original forest grid
        forest_grid, _, _ = plant_tree(forest_grid, optimal_tree__cost[0]) 
        empty_idxs.remove(list(optimal_tree__cost[0])) # removing this tree from the list of empty spaces


        # metric calculations
        tree_ct += 1
        forest_density = tree_ct/(L**2)
        
        # calculating timber yield (accounting for the expected fire size)
        timber_yield = forest_density - (optimal_tree__cost[1]/L**2)

        
        # saving values of interest 
        epoch2forest_grid[tree_ct] = forest_grid # note epochs are based at one here
        epoch2cost[tree_ct] = optimal_tree__cost[1]
        epoch2yield[tree_ct] = timber_yield
        
        completion.update(1)
    completion.close()
    
    
    # parsing run results for items of interest to return
    # the epoch with the greatest timber yield after expected fire size
    optimal_epoch = max(epoch2yield.items(), key=lambda x: x[1])[0]
    optimal_forest = epoch2forest_grid[optimal_epoch]
    
    
    return epoch2forest_grid, optimal_forest, epoch2yield