import matplotlib.pyplot as plt
from hubpersistence.read_data_from_csv import read_graph_structure_from_csv
from hubpersistence.weighted_graph import WeightedGraph

path_to_csv = '/Users/mattiagiuseppebergomi/Desktop/perscomb/code/cornerpoint_selection/data/literature/GOT/all.csv'
graph_structure = read_graph_structure_from_csv(path_to_csv)
graph = WeightedGraph(graph_structure)
graph.build_graph()
graph.build_filtered_subgraphs()
graph.get_temporary_hubs_along_filtration()
graph.steady_hubs_persistence()
graph.ranging_hubs_persistence()
graph.plot_steady_persistence_diagram(export_json=True)
graph.plot_ranging_persistence_diagram(export_json=True)
plt.show()
