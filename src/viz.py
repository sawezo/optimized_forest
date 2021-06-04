"""
visualization functions
"""

# imports
import numpy as np


# functions
def increment_tree_cell_values(X, color_ct=5):
    """
    adding color variety to cells where there are trees
    intended for use with matrix of zeros and ones and outputs 
    matrix where some of the ones are other non-zero integers
    """
    trees_per_color = len(np.argwhere(X==1))//color_ct # each non-zero integer will have same count (barring rounding)
 
    
    for value_increment in list(range(color_ct)):
        
        tree_idxs = np.argwhere(X==1) # indices of the trees
        if len(tree_idxs) == 0: # first pass for time visualization 
            continue
        
        try:
            tree_idx_idxs = np.random.choice(list(range(len(tree_idxs))), trees_per_color, replace=False)
        except ValueError: # trying to choose more cells than left
            tree_idx_idxs = np.array(list(range(len(tree_idxs))))
            
        tree_idxs_to_change = tree_idxs[tree_idx_idxs]
        
        for idx in tree_idxs_to_change:
            X[idx[0], idx[1]] += value_increment # incrementing trees in this group
        
    
    return X
