from __future__ import absolute_import, division, print_function
from sys import platform
if platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")
import sys
sys.path.append('../../')
import matplotlib.pyplot as plt
from perscomb.read_data_from_csv import read_graph_structure_from_csv
from perscomb.weighted_graph import WeightedGraph

def opp(x):
    return -x

def identity(x):
    return x

path_to_csv = '../../data/neuroscience/celegans_connectome.csv'
graph_structure = read_graph_structure_from_csv(path_to_csv)
graph = WeightedGraph(graph_structure)
graph.build_graph()
graph.build_filtered_subgraphs(weight_transform =opp, sublevel = True)
graph.get_temporary_hubs_along_filtration()
graph.ranging_hubs_persistence()
graph.plot_ranging_persistence_diagram(export_json = './c_elegans')
