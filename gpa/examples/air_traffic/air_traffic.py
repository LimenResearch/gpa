import os
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from gpa.examples.air_traffic.read_data import *
from gpa.weighted_graph import WeightedGraph
from gpa.constants import STATIC_FOLDER

def reproduce_thesis_figure(graph_structure_number=2, above_max_diagonal_gap=True):
    path_to_csv_1 = os.path.join(STATIC_FOLDER, "aux", "air_traffic", "dist_mat.csv")
    path_to_csv_2 = os.path.join(STATIC_FOLDER, "aux", "air_traffic", "freq_mat.csv")
    graph_structure = read_csv_distance_matrix(path_to_csv_1)
    graph_structure_2 = read_csv_frequencies_matrix(path_to_csv_2)
    graph_structure_1 = get_structure(graph_structure, graph_structure_2,
                                      product=False)
    graph_structure_3 = get_structure(graph_structure_2, graph_structure,
                                       product=True)
    graph_structures = [graph_structure, graph_structure_1, graph_structure_2,
                        graph_structure_3]
    graph = WeightedGraph(graph_structures[graph_structure_number])
    graph.build_graph()
    graph.build_filtered_subgraphs(weight_transform=identity)
    graph.get_temporary_hubs_along_filtration()
    # Steady
    graph.steady_persistence(above_max_diagonal_gap=above_max_diagonal_gap,
                                  gap_number=0)
    fig, ax = plt.subplots()
    graph.steady_pd.plot_gudhi(ax,
            persistence_to_plot = graph.steady_pd.persistence_to_plot)
    if hasattr(graph.steady_pd, 'proper_cornerpoints_above_gap'):
        graph.steady_pd.plot_nth_widest_gap(ax_handle =ax, n = graph.gap_number)
        graph.steady_pd.mark_points_above_diagonal_gaps(ax)
    plt.show()
    print ('steady hubs above gap:', graph.steady_pd.proper_cornerpoints_above_gap)
    pprint([(c.vertex, c.persistence) for c in graph.steady_pd.proper_cornerpoints])
    # Ranging
    graph.ranging_persistence(above_max_diagonal_gap = above_max_diagonal_gap )
    fig, ax = plt.subplots()
    graph.steady_pd.plot_gudhi(ax,
            persistence_to_plot = graph.ranging_pd.persistence_to_plot)
    if hasattr(graph.ranging_pd, 'proper_cornerpoints_above_gap'):
        graph.ranging_pd.plot_nth_widest_gap(ax_handle =ax, n = graph.gap_number)
        graph.ranging_pd.mark_points_above_diagonal_gaps(ax)
    plt.show()
    print ('ranging hubs above gap:', graph.ranging_pd.proper_cornerpoints_above_gap)
    pprint([(c.vertex, c.persistence) for c in graph.ranging_pd.proper_cornerpoints])

def get_structure(gf1, gf2, product = False):
    graph_structure = []

    for j in gf1:
        for i in gf2:
            if i[0] == j[0] and i[1] == j[1]:
                if not product:
                    graph_structure.append((j[0], j[1], j[2]))
                else:
                    graph_structure.append((j[0], j[1], i[2] * j[2]))

    return graph_structure

def opp(array):
        return -array

def identity(array):
        return array

def max_(array):
        return np.max(array) - array

def normal(array):
    return array / np.linalg.norm(array)

if __name__ == "__main__":
    reproduce_thesis_figure(graph_structure_number = 1)
