from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../../')
from sys import platform
if platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from perscomb.read_data_from_csv import read_graph_structure_from_csv
from perscomb.weighted_graph import WeightedGraph

path_to_csv = '/Users/mattiagiuseppebergomi/Desktop/perscomb/code/cornerpoint_selection/data/fintech/soc-sign-bitcoinotc.csv'
graph_structure = read_graph_structure_from_csv(path_to_csv, sep = ",")
graph = WeightedGraph(graph_structure)
graph.build_graph()
graph.build_filtered_subgraphs()
graph.get_temporary_hubs_along_filtration()
graph.steady_hubs_persistence()
graph.ranging_hubs_persistence()
graph.plot_steady_persistence_diagram(show = True)
graph.plot_ranging_persistence_diagram(show = True)
plt.show()
