import matplotlib.pyplot as plt
from hubpersistence.read_data_from_csv import read_graph_structure_from_csv
from hubpersistence.weighted_graph import WeightedGraph
from hubpersistence.constants import DATA_FOLDER

path_to_csv = os.path.join(DATA_FOLDER, "literature", "GOT". "all.csv")
graph_structure = read_graph_structure_from_csv(path_to_csv)
graph = WeightedGraph(graph_structure)
graph.build_graph()
graph.build_filtered_subgraphs()
graph.get_temporary_hubs_along_filtration()
graph.steady_persistence()
graph.ranging_persistence()
graph.plot_persistence_diagram()
plt.show()
