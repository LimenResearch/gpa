from hubpersistence.read_data_from_csv import read_graph_structure_from_csv
from hubpersistence.weighted_graph import WeightedGraph
from hubpersistence.constants import DATA_FOLDER


def opp(x):
    return -x

def identity(x):
    return x


path_to_csv = os.path.join(DATA_FOLDER, "neuroscience", "celegans_connectome.csv")
graph_structure = read_graph_structure_from_csv(path_to_csv)
graph = WeightedGraph(graph_structure)
graph.build_graph()
graph.build_filtered_subgraphs(weight_transform =opp, sublevel = True)
graph.get_temporary_hubs_along_filtration()
graph.ranging_persistence()
graph.plot_persistence_diagram(export_json = './c_elegans')
