import matplotlib.pyplot as plt
import numpy as np
from gpa.read_data_from_csv import read_csv_distance_matrix
from gpa.weighted_graph import WeightedGraph
import seaborn as sns
sns.set()

def identity(array):
    return array


def minus(array):
    return -array

if __name__ == "__main__":
    from gpa.constants import DATA_FOLDER
    path_to_csv = '/Users/mattiagiuseppebergomi/Desktop/perscomb/code/cornerpoint_selection/data/transportation/Airports.csv'
    graph_structure = read_csv_distance_matrix(path_to_csv)

    graph = WeightedGraph(graph_structure)
    graph.build_graph()
    graph.build_filtered_subgraphs(weight_transform = identity)
    graph.get_temporary_hubs_along_filtration()
    graph.steady_hubs_persistence()
    graph.ranging_hubs_persistence()
    graph.plot_steady_persistence_diagram(show = True)
    graph.plot_ranging_persistence_diagram(show = True)
    plt.show()
