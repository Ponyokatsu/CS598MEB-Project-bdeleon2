from IPython.display import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import cassiopeia as cas
from cassiopeia.tools import fitch_hartigan, fitch_count
import networkx as nx
import seaborn as sns
from enumeration import fitch_hartigan_enumeration, fitch_hartigan_print,fitch_hartigan_enumeration_restricted, sankoff_enumeration
import os

def build_matrices(clone_num,prt=1):
    tissues = ['LL', "RE", 'RW', "M1", "M2", "Liv"]
    tissues_idx = {}
    for i,t in enumerate(tissues):
        tissues_idx[t] = i
    tree_paths = ["/home/chunke/cs598MEB/trees/m5k_lg"+ str(i) +"_tree_hybrid_priors.alleleThresh.processed.txt" for i in range(1,101)]
    tree_path = tree_paths[clone_num -1]
    min_print(tree_path,prt)
    directory = "output_trees_" + str(clone_num)
    parent_dir = "/home/chunke/cs598MEB/MetastasisTracing-master/"
    path = os.path.join(parent_dir, directory) 
    try:  
        os.mkdir(path)  
    except OSError as error:  
        min_print("directory already exists",prt)
    output_trees_path = "output_trees_{0}/tree_".format(clone_num)
    newick = ""
    with open(tree_path, 'r') as file:
        newick = file.read()
    if(len(newick) > 0):
        cas_tree = cas.data.CassiopeiaTree()
        cas_tree.populate_tree(newick)
        leaves = [leaf for leaf in cas_tree.leaves]
        meta_data = [[leaf.split('.')[0]] for leaf in leaves]
        cas_tree.cell_meta = pd.DataFrame(meta_data,columns=['tissue'],index=leaves)
        # fig = cas.pl.plot_plotly(cas_tree)
        # fig.show()
        # labeled_tree = fitch_hartigan(cas_tree,'tissue',label_key='t_label',copy=True)
        min_print("beginning enumeration",prt)
        enumerated_trees,_,_,min_comigration_trees,min_comigration_dicts= fitch_hartigan_enumeration(cas_tree,'tissue',label_key='t_label')
        transition_matrix = np.zeros((len(tissues),len(tissues)))
        for comigration_dict in min_comigration_dicts: # build transition matrix
            for mgr,cnt in comigration_dict.items():
                mgr = mgr.split("->")
                src = tissues_idx[mgr[0]]
                dst = tissues_idx[mgr[1]]
                transition_matrix[src,dst] += cnt
        
        transition_matrix = pd.DataFrame(transition_matrix,index=tissues,columns=tissues)
        fc_matrix = fitch_count(cas_tree,'tissue',unique_states=tissues)
        np.fill_diagonal(fc_matrix.values, 0)
        min_print("raw Fitch-Hartigan enumeration transition matrix",prt)
        min_print(transition_matrix,prt)
        min_print("raw FitchCount transition matrix",prt)
        min_print(fc_matrix,prt)
        # fc_matrix = (fc_matrix/fc_matrix.max(axis=None)).round(3)
        fc_matrix = fc_matrix.apply(lambda x: x / max(1, x.sum()), axis=1)
        # transition_matrix = (transition_matrix/transition_matrix.max(axis=None)).round(3)
        transition_matrix = transition_matrix.apply(lambda x: x / max(1, x.sum()), axis=1)
        min_print("normalized Fitch-Hartigan enumeration transition matrix",prt)
        min_print(transition_matrix,prt)
        min_print("normalized FitchCount transition matrix",prt)
        min_print(fc_matrix,prt)
        min_print("beginning writing to files",prt)
        for idx,tree in enumerate(enumerated_trees):
            tree_labeling = ""
            # produces a preorder DFS of the tree 
            for node in tree.depth_first_traverse_nodes(source=tree.root,postorder=False):
                tree_labeling += node + ":" + tree.get_attribute(node,'t_label') + "\n"
            out_path = output_trees_path + str(idx) + ".txt"
            with open(out_path,'w') as file:
                file.write(tree_labeling)
        # to read it back in, take the tree, make a list of copies, then for each line set_attribute the node
        return transition_matrix,fc_matrix


def build_matrices_restricted(clone_num,prt=1):
    tissues = ['LL', "RE", 'RW', "M1", "M2", "Liv"]
    tissues_idx = {}
    for i,t in enumerate(tissues):
        tissues_idx[t] = i
    tree_paths = ["/home/chunke/cs598MEB/trees/m5k_lg"+ str(i) +"_tree_hybrid_priors.alleleThresh.processed.txt" for i in range(1,101)]
    tree_path = tree_paths[clone_num -1]
    min_print(tree_path,prt)
    directory = "output_trees_r_" + str(clone_num)
    parent_dir = "/home/chunke/cs598MEB/MetastasisTracing-master/"
    path = os.path.join(parent_dir, directory) 
    try:  
        os.mkdir(path)  
    except OSError as error:  
        min_print("directory already exists",prt)
    output_trees_path = "output_trees_r_{0}/tree_".format(clone_num)
    newick = ""
    with open(tree_path, 'r') as file:
        newick = file.read()
    if(len(newick) > 0):
        cas_tree = cas.data.CassiopeiaTree()
        cas_tree.populate_tree(newick)
        leaves = [leaf for leaf in cas_tree.leaves]
        meta_data = [[leaf.split('.')[0]] for leaf in leaves]
        cas_tree.cell_meta = pd.DataFrame(meta_data,columns=['tissue'],index=leaves)
        # fig = cas.pl.plot_plotly(cas_tree)
        # fig.show()
        min_print("beginning enumeration",prt)
        enumerated_trees,_,_,min_comigration_trees,min_comigration_dicts= fitch_hartigan_enumeration_restricted(cas_tree,'tissue',label_key='t_label')
        transition_matrix = np.zeros((len(tissues),len(tissues)))
        for comigration_dict in min_comigration_dicts: # build transition matrix
            for mgr,cnt in comigration_dict.items():
                mgr = mgr.split("->")
                src = tissues_idx[mgr[0]]
                dst = tissues_idx[mgr[1]]
                transition_matrix[src,dst] += cnt
        
        transition_matrix = pd.DataFrame(transition_matrix,index=tissues,columns=tissues)
        min_print("raw restricted Fitch-Hartigan enumeration transition matrix",prt)
        min_print(transition_matrix,prt)
        transition_matrix = transition_matrix.apply(lambda x: x / max(1, x.sum()), axis=1)
        min_print("normalized restricted Fitch-Hartigan enumeration transition matrix",prt)
        min_print(transition_matrix,prt)
       
        min_print("beginning writing to files",prt)
        for idx,tree in enumerate(enumerated_trees):
            tree_labeling = ""
            # produces a preorder DFS of the tree 
            for node in tree.depth_first_traverse_nodes(source=tree.root,postorder=False):
                tree_labeling += node + ":" + tree.get_attribute(node,'t_label') + "\n"
            out_path = output_trees_path + str(idx) + ".txt"
            with open(out_path,'w') as file:
                file.write(tree_labeling)
        # to read it back in, take the tree, make a list of copies, then for each line set_attribute the node
        return transition_matrix


def build_matrices_sankoff(clone_num,prt=1):
    tissues = ['LL', "RE", 'RW', "M1", "M2", "Liv"]
    tissues_idx = {}
    for i,t in enumerate(tissues):
        tissues_idx[t] = i
    tree_paths = ["/home/chunke/cs598MEB/trees/m5k_lg"+ str(i) +"_tree_hybrid_priors.alleleThresh.processed.txt" for i in range(1,101)]
    tree_path = tree_paths[clone_num -1]
    min_print(tree_path,prt)
    directory = "output_trees_s_" + str(clone_num)
    parent_dir = "/home/chunke/cs598MEB/MetastasisTracing-master/"
    path = os.path.join(parent_dir, directory) 
    try:  
        os.mkdir(path)  
    except OSError as error:  
        min_print("directory already exists",prt)
    output_trees_path = "output_trees_s_{0}/tree_".format(clone_num)
    newick = ""
    with open(tree_path, 'r') as file:
        newick = file.read()
    if(len(newick) > 0):
        cas_tree = cas.data.CassiopeiaTree()
        cas_tree.populate_tree(newick)
        leaves = [leaf for leaf in cas_tree.leaves]
        meta_data = [[leaf.split('.')[0]] for leaf in leaves]
        cas_tree.cell_meta = pd.DataFrame(meta_data,columns=['tissue'],index=leaves)
        # fig = cas.pl.plot_plotly(cas_tree)
        # fig.show()
        min_print("beginning enumeration",prt)
        enumerated_trees,_,_,min_comigration_trees,min_comigration_dicts = sankoff_enumeration(cas_tree,'tissue',label_key='t_label',tissues=tissues,prim_site='LL')
        transition_matrix = np.zeros((len(tissues),len(tissues)))
        for comigration_dict in min_comigration_dicts: # build transition matrix
            for mgr,cnt in comigration_dict.items():
                mgr = mgr.split("->")
                src = tissues_idx[mgr[0]]
                dst = tissues_idx[mgr[1]]
                transition_matrix[src,dst] += cnt
        
        transition_matrix = pd.DataFrame(transition_matrix,index=tissues,columns=tissues)
    
        min_print("raw Sankoff enumeration transition matrix",prt)
        min_print(transition_matrix,prt)
        # transition_matrix = (transition_matrix/transition_matrix.max(axis=None)).round(3)
        transition_matrix = transition_matrix.apply(lambda x: x / max(1, x.sum()), axis=1)
        min_print("normalized Sankoff enumeration transition matrix",prt)
        min_print(transition_matrix,prt)
        min_print("beginning writing to files",prt)
        for idx,tree in enumerate(enumerated_trees):
            tree_labeling = ""
            # produces a preorder DFS of the tree 
            for node in tree.depth_first_traverse_nodes(source=tree.root,postorder=False):
                tree_labeling += node + ":" + tree.get_attribute(node,'t_label') + "\n"
            out_path = output_trees_path + str(idx) + ".txt"
            with open(out_path,'w') as file:
                file.write(tree_labeling)
        # to read it back in, take the tree, make a list of copies, then for each line set_attribute the node
        return transition_matrix

def min_print(str,prt=1):
    if(prt):
        print(str)
def build_matrices_fc(clone_num,prt=1):
    tissues = ['LL', "RE", 'RW', "M1", "M2", "Liv"]
    tissues_idx = {}
    for i,t in enumerate(tissues):
        tissues_idx[t] = i
    tree_paths = ["/home/chunke/cs598MEB/trees/m5k_lg"+ str(i) +"_tree_hybrid_priors.alleleThresh.processed.txt" for i in range(1,101)]
    tree_path = tree_paths[clone_num -1]
    min_print(tree_path,prt)
    directory = "output_trees_" + str(clone_num)
    parent_dir = "/home/chunke/cs598MEB/MetastasisTracing-master/"
    path = os.path.join(parent_dir, directory) 
    try:  
        os.mkdir(path)  
    except OSError as error:  
        min_print("directory already exists",prt)
    output_trees_path = "output_trees_{0}/tree_".format(clone_num)
    newick = ""
    with open(tree_path, 'r') as file:
        newick = file.read()
    if(len(newick) > 0):
        cas_tree = cas.data.CassiopeiaTree()
        cas_tree.populate_tree(newick)
        leaves = [leaf for leaf in cas_tree.leaves]
        meta_data = [[leaf.split('.')[0]] for leaf in leaves]
        cas_tree.cell_meta = pd.DataFrame(meta_data,columns=['tissue'],index=leaves)
        fc_matrix = fitch_count(cas_tree,'tissue',unique_states=tissues)
        np.fill_diagonal(fc_matrix.values, 0)
        fc_matrix = fc_matrix.apply(lambda x: x / max(1, x.sum()), axis=1)
        # to read it back in, take the tree, make a list of copies, then for each line set_attribute the node
        return fc_matrix

def number_of_nodes(clone_num,prt=1):
    tissues = ['LL', "RE", 'RW', "M1", "M2", "Liv"]
    tissues_idx = {}
    for i,t in enumerate(tissues):
        tissues_idx[t] = i
    tree_paths = ["/home/chunke/cs598MEB/trees/m5k_lg"+ str(i) +"_tree_hybrid_priors.alleleThresh.processed.txt" for i in range(1,101)]
    tree_path = tree_paths[clone_num -1]
    min_print(tree_path,prt)
    newick = ""
    with open(tree_path, 'r') as file:
        newick = file.read()
    if(len(newick) > 0):
        cas_tree = cas.data.CassiopeiaTree()
        cas_tree.populate_tree(newick)
        leaves = [leaf for leaf in cas_tree.leaves]
        meta_data = [[leaf.split('.')[0]] for leaf in leaves]
        cas_tree.cell_meta = pd.DataFrame(meta_data,columns=['tissue'],index=leaves)
        print("Nodes",len(cas_tree.nodes))
        return len(cas_tree.nodes)
clones = [64,43,21]
node_len = []
for c in clones:
    print("Clone",c)
    node_len.append(number_of_nodes(c,0))
print("Node mean",sum(node_len)/len(node_len))
print("Node min",min(node_len))
print("Node max", max(node_len))