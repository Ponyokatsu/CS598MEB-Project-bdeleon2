from IPython.display import Image

import numpy as np
import pandas as pd

import cassiopeia as cas
import networkx as nx
from typing import Dict, List, Optional

import itertools
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_numeric_dtype


from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins.errors import (
    CassiopeiaError,
    CassiopeiaTreeError,
    FitchCountError,
)


def fitch_hartigan_print(
    cassiopeia_tree: CassiopeiaTree,
    meta_item: str,
    root: Optional[str] = None,
    state_key: str = "S1",
    label_key: str = "label",
    copy: bool = False,
) -> Optional[CassiopeiaTree]:
    """Run the Fitch-Hartigan algorithm.
    
    Performs the full Fitch-Hartigan small parsimony algorithm which, given
    a set of states for the leaves, infers the most-parsimonious set of states
    and returns a random solution that satisfies the maximum-parsimony
    criterion. The solution will be stored in the label key specified by the
    user (by default 'label'). This function will modify the tree in place
    if `copy=False`.

    Args:
        cassiopeia_tree: CassiopeiaTree that has been processed with the
            Fitch-Hartigan bottom-up algorithm.
        meta_item: A column in the CassiopeiaTree cell meta corresponding to a
            categorical variable.
        root: Root from which to begin this refinement. Only the subtree below
            this node will be considered.
        state_key: Attribute key that stores the Fitch-Hartigan ancestral
            states.
        label_key: Key to add that stores the maximum-parsimony assignment
            inferred from the Fitch-Hartigan top-down refinement.
        copy: Modify the tree in place or not.
    
    Returns:
        A new CassiopeiaTree if the copy is set to True, else None.
    """

    cassiopeia_tree = cassiopeia_tree.copy() if copy else cassiopeia_tree

    fitch_hartigan_bottom_up(cassiopeia_tree, meta_item, state_key)

    fitch_hartigan_top_down(cassiopeia_tree, root, state_key, label_key)

    return cassiopeia_tree if copy else None

def fitch_hartigan_bottom_up(
    cassiopeia_tree: CassiopeiaTree,
    meta_item: str,
    add_key: str = "S1",
    copy: bool = False,
) -> Optional[CassiopeiaTree]:
    """Performs Fitch-Hartigan bottom-up ancestral reconstruction.

    Performs the bottom-up phase of the Fitch-Hartigan small parsimony
    algorithm. A new attribute called "S1" will be added to each node
    storing the optimal set of ancestral states inferred from this bottom-up 
    algorithm. If copy is False, the tree will be modified in place.
     

    Args:
        cassiopeia_tree: CassiopeiaTree object with cell meta data.
        meta_item: A column in the CassiopeiaTree cell meta corresponding to a
            categorical variable.
        add_key: Key to add for bottom-up reconstruction
        copy: Modify the tree in place or not.

    Returns:
        A new CassiopeiaTree if the copy is set to True, else None.

    Raises:
        CassiopeiaError if the tree does not have the specified meta data
            or the meta data is not categorical.
    """

    if meta_item not in cassiopeia_tree.cell_meta.columns:
        raise CassiopeiaError("Meta item does not exist in the cassiopeia tree")

    meta = cassiopeia_tree.cell_meta[meta_item]

    if is_numeric_dtype(meta):
        raise CassiopeiaError("Meta item is not a categorical variable.")

    if not is_categorical_dtype(meta):
        meta = meta.astype("category")

    cassiopeia_tree = cassiopeia_tree.copy() if copy else cassiopeia_tree
    
    for node in cassiopeia_tree.depth_first_traverse_nodes():

        if cassiopeia_tree.is_leaf(node):
            cassiopeia_tree.set_attribute(node, add_key, [meta.loc[node]])

        else:
            children = cassiopeia_tree.children(node)
            if len(children) == 1:
                # list_of_children = []
                # while(1):
                #     if children[0] in list_of_children:
                #         print("ruh roh raggy")
                #     list_of_children.append(children[0])
                #     children = cassiopeia_tree.children(children[0])
                    
                #     if cassiopeia_tree.is_leaf(children[0]):
                #         print(cassiopeia_tree.get_attribute(children[0],add_key))
                #         break

                child_assignment = cassiopeia_tree.get_attribute(
                    children[0], add_key
                )
                cassiopeia_tree.set_attribute(node, add_key, [child_assignment])

            all_labels = np.concatenate(
                [
                    cassiopeia_tree.get_attribute(child, add_key)
                    for child in children
                ]
            )
            states, frequencies = np.unique(all_labels, return_counts=True)

            S1 = states[np.where(frequencies == np.max(frequencies))]
            cassiopeia_tree.set_attribute(node, add_key, S1)

    return cassiopeia_tree if copy else None


def fitch_hartigan_top_down(
    cassiopeia_tree: CassiopeiaTree,
    root: Optional[str] = None,
    state_key: str = "S1",
    label_key: str = "label",
    copy: bool = False,
) -> Optional[CassiopeiaTree]:
    """Run Fitch-Hartigan top-down refinement

    Runs the Fitch-Hartigan top-down algorithm which selects an optimal solution
    from the tree rooted at the specified root.

    Args:
        cassiopeia_tree: CassiopeiaTree that has been processed with the
            Fitch-Hartigan bottom-up algorithm.
        root: Root from which to begin this refinement. Only the subtree below
            this node will be considered.
        state_key: Attribute key that stores the Fitch-Hartigan ancestral
            states.
        label_key: Key to add that stores the maximum-parsimony assignment
            inferred from the Fitch-Hartigan top-down refinement.
        copy: Modify the tree in place or not.

    Returns:
        A new CassiopeiaTree if the copy is set to True, else None.

    Raises:
        A CassiopeiaTreeError if Fitch-Hartigan bottom-up has not been called
        or if the state_key does not exist for a node.
    """

    # assign root
    root = cassiopeia_tree.root if (root is None) else root

    cassiopeia_tree = cassiopeia_tree.copy() if copy else cassiopeia_tree

    for node in cassiopeia_tree.depth_first_traverse_nodes(
        source=root, postorder=False
    ):

        if node == root:
            root_states = cassiopeia_tree.get_attribute(root, state_key)
            cassiopeia_tree.set_attribute(
                root, label_key, np.random.choice(root_states)
            )
            print("root choices:",len(root_states))
            continue

        parent = cassiopeia_tree.parent(node)
        parent_label = cassiopeia_tree.get_attribute(parent, label_key)
        optimal_node_states = cassiopeia_tree.get_attribute(node, state_key)

        if parent_label in optimal_node_states:
            cassiopeia_tree.set_attribute(node, label_key, parent_label)

        else:
            cassiopeia_tree.set_attribute(
                node, label_key, np.random.choice(optimal_node_states)
            )
            print("optimal state choices:",len(optimal_node_states))

    return cassiopeia_tree if copy else None

# change to accept a primary site, if no primary site is in optimal it's broke
def fitch_hart_enum_top_down(
    tree: CassiopeiaTree,
    state_key: str = "S1",
    label_key: str = "label",
) -> list:
    trees = [tree.copy()]
    comigrations = [{}]
    root = tree.root
    for node in tree.depth_first_traverse_nodes(source=root,postorder=False):
        if node == root:
            root_states = tree.get_attribute(root,state_key) # one tree rn since we are at the root
            n_states = len(root_states)
            # print("root choices:",n_states)
            trees[0].set_attribute(root,label_key,root_states[0]) # whether there are multiple options or not, set the current copy (the only one) to the first available state
            if(n_states > 1): # we have to add copies  
                new_trees = []
                new_comigrations = []
                for s in range(1,n_states): # loop thru all states
                    new_tree = tree.copy() # in a normal node, we have to loop thru all trees and make copies, but in the root there is only one tree 
                    new_tree.set_attribute(root, label_key, root_states[s])
                    new_trees.append(new_tree)
                    new_comigrations.append({})
                trees += new_trees
                comigrations += new_comigrations
            continue
        
        parent = tree.parent(node) # parent is the same regardless of copy
        new_trees = []
        new_comigrations = []
        for t in range(len(trees)): 
            parent_label = trees[t].get_attribute(parent, label_key)
            optimal_node_states = trees[t].get_attribute(node, state_key)
            n_states = len(optimal_node_states)
            if parent_label in optimal_node_states:
                trees[t].set_attribute(node, label_key, parent_label) # in this specific tree, there is no need to add copies
            else:
                # print("tree",t,"optimal choices:",n_states)
                # there is a migration so check to add to comigration dictionary
                if(n_states > 1):
                    for s in range(1,n_states):
                        new_tree = trees[t].copy()
                        new_comigration = comigrations[t].copy()
                        new_tree.set_attribute(node,label_key,optimal_node_states[s])
                        new_trees.append(new_tree)
                        migration = parent_label + "->" + optimal_node_states[s]
                        if migration in new_comigration: # count the total number of migrations
                            new_comigration[migration] += 1
                        else:
                            new_comigration[migration] = 1
                        new_comigrations.append(new_comigration)
                trees[t].set_attribute(node,label_key,optimal_node_states[0]) # whether there are multiple options or not, set the current copy to the first available state
                migration = parent_label + "->" + optimal_node_states[0]
                if migration in comigrations[t]:
                    comigrations[t][migration] += 1
                else:
                    comigrations[t][migration] = 1
        trees += new_trees
        comigrations += new_comigrations
    print("Number of trees:",len(trees))
    min_comigration = len(comigrations[0])
    for c_dict in comigrations:
        if len(c_dict) < min_comigration:
            min_comigration = len(c_dict)
    min_comigration_trees = [] # the trees that have min comigrations
    min_comigration_dicts = [] # the corresponding comigration dicts
    for c in range(len(comigrations)):
        if len(comigrations[c]) == min_comigration:
            min_comigration_trees.append(trees[c])
            min_comigration_dicts.append(comigrations[c])
    print("Min comigrations:",min_comigration)
    print("Number of trees that obey min comigrations",len(min_comigration_trees))
    return trees, comigrations, min_comigration, min_comigration_trees,min_comigration_dicts

# change to accept primary site
def fitch_hartigan_enumeration(
    tree: CassiopeiaTree,
    meta_item: str,
    state_key: str = "S1",
    label_key: str = "label"
) -> list:
    
    tree = fitch_hartigan_bottom_up(tree, meta_item, state_key,copy=True) 

    return fitch_hart_enum_top_down(tree, state_key, label_key)
                    
    
def fitch_hart_enum_top_down_restricted(
    tree: CassiopeiaTree,
    state_key: str = "S1",
    label_key: str = "label",
    prim_site: str = "LL"
) -> list:
    trees = [tree.copy()]
    comigrations = [{}]
    root = tree.root
    for node in tree.depth_first_traverse_nodes(source=root,postorder=False):
        if node == root:
            trees[0].set_attribute(root,label_key,prim_site) # set root to primary site
            continue
        
        parent = tree.parent(node) # parent is the same regardless of copy
        new_trees = []
        new_comigrations = []
        for t in range(len(trees)): 
            parent_label = trees[t].get_attribute(parent, label_key)
            optimal_node_states = trees[t].get_attribute(node, state_key)
            n_states = len(optimal_node_states)
            if parent_label in optimal_node_states:
                trees[t].set_attribute(node, label_key, parent_label) # in this specific tree, there is no need to add copies
            else:
                # print("tree",t,"optimal choices:",n_states)
                # there is a migration so check to add to comigration dictionary
                if(n_states > 1):
                    for s in range(1,n_states):
                        new_tree = trees[t].copy()
                        new_comigration = comigrations[t].copy()
                        new_tree.set_attribute(node,label_key,optimal_node_states[s])
                        new_trees.append(new_tree)
                        migration = parent_label + "->" + optimal_node_states[s]
                        if migration in new_comigration: # count the total number of migrations
                            new_comigration[migration] += 1
                        else:
                            new_comigration[migration] = 1
                        new_comigrations.append(new_comigration)
                trees[t].set_attribute(node,label_key,optimal_node_states[0]) # whether there are multiple options or not, set the current copy to the first available state
                migration = parent_label + "->" + optimal_node_states[0]
                if migration in comigrations[t]:
                    comigrations[t][migration] += 1
                else:
                    comigrations[t][migration] = 1
        trees += new_trees
        comigrations += new_comigrations
    print("Number of trees:",len(trees))
    min_comigration = len(comigrations[0])
    for c_dict in comigrations:
        if len(c_dict) < min_comigration:
            min_comigration = len(c_dict)
    min_comigration_trees = [] # the trees that have min comigrations
    min_comigration_dicts = [] # the corresponding comigration dicts
    for c in range(len(comigrations)):
        if len(comigrations[c]) == min_comigration:
            min_comigration_trees.append(trees[c])
            min_comigration_dicts.append(comigrations[c])
    print("Min comigrations:",min_comigration)
    print("Number of trees that obey min comigrations",len(min_comigration_trees))
    return trees, comigrations, min_comigration, min_comigration_trees,min_comigration_dicts


def fitch_hartigan_enumeration_restricted(
    tree: CassiopeiaTree,
    meta_item: str,
    state_key: str = "S1",
    label_key: str = "label",
    prim_site: str = "LL"
) -> list:
    
    tree = fitch_hartigan_bottom_up(tree, meta_item, state_key,copy=True) 

    return fitch_hart_enum_top_down_restricted(tree, state_key, label_key,prim_site)


def sankoff_enumeration(
        tree: CassiopeiaTree,
        meta_item: str,
        state_key: str = "S1",
        label_key: str = "label",
        tissues: list = ['LL', "RE", 'RW', "M1", "M2", "Liv"],
        prim_site: str = 'LL'
) -> list:
    
    if meta_item not in tree.cell_meta.columns:
        raise CassiopeiaError("Meta item does not exist in the cassiopeia tree")

    meta = tree.cell_meta[meta_item]

    if is_numeric_dtype(meta):
        raise CassiopeiaError("Meta item is not a categorical variable.")

    if not is_categorical_dtype(meta):
        meta = meta.astype("category")

    tree = tree.copy()
    M = {}
    delta = {}
    SOLVE(tree,tissues,meta,M,tree.root,delta)
    return sankoff_enum_top_down(tree,delta,label_key,prim_site)
    

def SOLVE(tree,sigma,l,M,u,delta):
    if tree.is_leaf(u):
        for s in sigma:
            if l.loc[u] == s: # no cost to keeping the state of the leaf, 2nd recurrence
                if u in M:
                    M[u][s] = 0
                    delta[u][s] = []
                else:
                    M[u] = {s : 0}
                    delta[u] = {s : []}
            else: # infinite cost to changing the state of the leaf, 1st recurrence
                if u in M:
                    M[u][s] = float('inf')
                    delta[u][s] = []
                else:
                    M[u] = {s : float('inf')}
                    delta[u] = {s : []}
    else: 
        children = tree.children(u)
        for v in children: # recurse on children first
            SOLVE(tree,sigma,l,M,v,delta)
        for s in sigma: 
            if u in M:
                M[u][s] = 0
            else:
                M[u] = {s : 0}
            if u in delta:
                delta[u][s] = []
            else:
                delta[u] = {s : []}
            for v in children:# 3rd recurrence
                c = float('inf')
                for t in sigma: # finding the min cost
                    nc = COST(M,s,v,t)
                    if nc < c:
                        c = nc
                M[u][s] += c
                for t in sigma:
                    if(COST(M,s,v,t) == c):
                        if (v,t) not in delta[u][s]:
                            delta[u][s].append((v,t)) 

                    
def COST(M,s,v,t):
    cst = 0 if s == t else 1
    return cst + M[v][t]     

def sankoff_enum_top_down(tree, delta,label_key = "label", prim_site = "LL"):
    trees = [tree.copy()]
    comigrations = [{}]
    root = tree.root
    for node in tree.depth_first_traverse_nodes(source=root,postorder=False):
        if node == root:
            trees[0].set_attribute(root,label_key,prim_site) # set root to primary site
            continue
        parent = tree.parent(node) # parent is the same regardless of copy
        new_trees = []
        new_comigrations = []
        for t in range(len(trees)): 
            parent_label = trees[t].get_attribute(parent, label_key)
            children_labels = delta[parent][parent_label] # [(w,t),...] includes other children as well
            node_labels = []
            for w,cl in children_labels: 
                if w == node:
                    node_labels.append(cl) # only maintain this nodes possible labels
            n_states = len(node_labels)
            if(n_states > 1):
                for s in range(1,n_states):
                    new_tree = trees[t].copy()
                    new_comigration = comigrations[t].copy()
                    new_tree.set_attribute(node,label_key,node_labels[s])
                    new_trees.append(new_tree)
                    if parent_label != node_labels[s]:
                        migration = parent_label + "->" + node_labels[s]
                        if migration in new_comigration: # count the total number of migrations
                            new_comigration[migration] += 1
                        else:
                            new_comigration[migration] = 1
                    new_comigrations.append(new_comigration)
            trees[t].set_attribute(node,label_key,node_labels[0]) # whether there are multiple options or not, set the current copy to the first available state
            if parent_label != node_labels[0]:
                migration = parent_label + "->" + node_labels[0]
                if migration in comigrations[t]:
                    comigrations[t][migration] += 1
                else:
                    comigrations[t][migration] = 1
        trees += new_trees
        comigrations += new_comigrations
    print("Number of trees:",len(trees))
    min_comigration = len(comigrations[0])
    for c_dict in comigrations:
        if len(c_dict) < min_comigration:
            min_comigration = len(c_dict)
    min_comigration_trees = [] # the trees that have min comigrations
    min_comigration_dicts = [] # the corresponding comigration dicts
    for c in range(len(comigrations)):
        if len(comigrations[c]) == min_comigration:
            min_comigration_trees.append(trees[c])
            min_comigration_dicts.append(comigrations[c])
    print("Min comigrations:",min_comigration)
    print("Number of trees that obey min comigrations",len(min_comigration_trees))
    return trees, comigrations, min_comigration, min_comigration_trees,min_comigration_dicts