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
from scipy.special import rel_entr

def venn_diagram(num,algo1="r_",algo2="s_",algo3="",comig=False):
    algos = ["","r_","s_"]
    algos = [algo1,algo2,algo3]
    i = 0
    trees123 = [[],[],[]]
    # print(path1,path2)
    co = ""
    if comig:
        co = "mc_"
    for idx,algo in enumerate(algos):
        i = 0
        path = co + "output_trees_" + algo + str(num) + "/tree_"
        while(1):
            try:
                # open file
                tree = ""
                with open(path + str(i) + ".txt","r") as file:
                    lines = file.readlines()
                    for line in lines:
                        label = line.split(':')[1]
                        tree += label
                trees123[idx].append(tree)
            except:
                break  
            i += 1
    
  
    trees1 = set(trees123[0])
    trees2 = set(trees123[1])
    trees3 = set(trees123[2])
    translate = {"r_" : "Restricted", "s_" : "Sankoff", "" : "Vanilla"}
    print(translate[algo1],"only",len(trees1-trees2-trees3))
    print(translate[algo2],"only",len(trees2-trees1-trees3))
    print(translate[algo3],"only",len(trees3-trees1-trees2))
    print(translate[algo1],"and",translate[algo2],len((trees1 & trees2)-trees3))
    print(translate[algo1],"and",translate[algo3],len((trees1 & trees3)-trees2))
    print(translate[algo2],"and",translate[algo3],len((trees2 & trees3)-trees1))
    print(translate[algo1],"and",translate[algo2],"and",translate[algo3],len(trees1 & trees2 & trees3))


def kl_divergence(a,b):
    a += 1e-6
    b += 1e-6
    a = a.apply(lambda x: x / max(1, x.sum()), axis=1)
    b = b.apply(lambda x: x / max(1, x.sum()), axis=1)
    a = a.to_numpy()
    b = b.to_numpy()
    kl_a = np.zeros(a.shape[0])
    kl_b = np.zeros(a.shape[0])
    for j in range(len(a)):
        kl_a[j] = sum(rel_entr(a[j],b[j]))
        kl_b[j] = sum(rel_entr(b[j],a[j]))

    return (kl_a.mean() + kl_b.mean())/2
