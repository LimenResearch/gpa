from __future__ import absolute_import, print_function, division
from sys import platform
if platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")
import networkx as nx
from networkx.algorithms.approximation import min_weighted_dominating_set
# import gudhi as gd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import json
from utils import grouped
from perscomb.persistence_diagram import PersistenceDiagram, CornerPoint

class WeightedGraph(object):
    """
    Attributes
    ----------

    weighted_edges : list
        list of tuples of the form [(v1, v2, w12), ...] where v1 and v2 are the
        labels associated to the vertices on the boundary of the edge and w12
        is its weight.
    """
    def __init__(self, weighted_edges):
        self.weighted_edges = weighted_edges

    def build_graph(self):
        """Builds the graph defined by the collection of weighted edges
        """
        self.G = nx.Graph()

        for v1, v2, w12 in self.weighted_edges:
            self.G.add_edge(v1, v2, weight = w12)

    def get_edge_filtration_values(self, subgraph = None,
                                    weight_transform = None):
        """Creates list of nodes and edges. Applies filtrating function to the
        weights of the edges and stores the values of the filtrating function
        in an array according to the ordering used to sort the edges in networkx
        edges dictionary
        """
        if subgraph is None:
            subgraph = self.G
        self.nodes = list(subgraph.nodes)
        self.edges = nx.get_edge_attributes(subgraph,'weight').keys()
        self.evaluate_weight_transform_and_set_on_edges(subgraph,
                                                            weight_transform)

    def evaluate_weight_transform_and_set_on_edges(self, subgraph,
                                                        weight_transform):
        """Creates a dictionary edge: value_of_the_weight_transform.

        Notes
        -----
        Does not use nx.set_edge_attributes. To be updated after networkx bug
        correction.
        See link_to_the_reported_issue
        """
        self.transformed_edges = self.get_filtration_values(subgraph,
                                                            weight_transform)
        self.transformed_edges_dict = {edge: value
                                            for edge, value in
                                            zip(self.edges, self.transformed_edges)}

    @staticmethod
    def weight_inverse(array):
        """Standard filtrating function
        """
        return 1 / array

    def get_filtration_values(self, subgraph, func):
        """Evaluates func on the weights defined on the edges
        """
        return func(np.asarray(nx.get_edge_attributes(subgraph,'weight').values()))


    def get_subgraph_edges(self, value, sublevel= True):
        """Returns the edges of self.G part of the sublevel set defined by value
        """
        if sublevel:
            return [edge + tuple([self.transformed_edges_dict[edge]])
                    for edge in self.transformed_edges_dict
                    if self.transformed_edges_dict[edge] <= value]
        else:
            return [edge + tuple([self.transformed_edges_dict[edge]])
                    for edge in self.transformed_edges_dict
                    if self.transformed_edges_dict[edge] >= value]

    @staticmethod
    def get_subgraph(edges):
        """Returns the subgraph defined by edges. Once the filtration is
        generated we are only interested in the 'hubbiness' of the nodes in the
        subgraph. Thus we do not set weights.
        """
        H = nx.Graph()
        [H.add_edge(v1, v2, weight =  np.round(w12, decimals = 2)) for v1, v2, w12 in edges]
        return H

    def build_filtered_subgraphs(self, weight_transform = None, sublevel = True):
        """Generates the filtration of G given the values of the filtrating
        function.
        """
        if weight_transform is None:
            weight_transform = self.weight_inverse

        self.weight_transform = weight_transform
        self.get_edge_filtration_values(weight_transform = weight_transform)
        self.filtration = []
        self.transformed_edges = np.unique(self.transformed_edges)
        self.transformed_edges.sort()

        for value in self.transformed_edges:
            edges = self.get_subgraph_edges(value, sublevel=sublevel)
            self.filtration.append(self.get_subgraph(edges))

    @staticmethod
    def get_number_of_neighbours(graph, node):
        """Returns the number of neighbours of a node
        """
        return len(graph[node])

    def is_t_hub(self, node = None, subgraph = None):
        """Given a node and a graph returns True if node is a temporary hub
        according to the definition given in [1]_

        .. [1] Link to the paper
        """
        if subgraph is None:
            subgraph = self.G
        degree_node = self.get_number_of_neighbours(subgraph, node)
        degrees_neighbouring_nodes = [self.get_number_of_neighbours(subgraph, neighbouring_node)
                                        for neighbouring_node in subgraph[node]]

        return max(degrees_neighbouring_nodes) < degree_node

    def get_t_hubs(self, subgraph = None):
        """Returns the list of the t_hubs in a subgraph. Typically subgraph is
        one of the subgraphs of the filtration of self.G
        """
        if subgraph is None:
            subgraph = self.G

        return [node for node in list(subgraph.nodes)
                    if self.is_t_hub(node = node, subgraph = subgraph)]

    def get_temporary_hubs_along_filtration(self):
        """Gets the nodes that are temporary hubs along the the filtration
        """
        self.filtered_t_hubs = {i : self.get_t_hubs(subgraph = g)
                                    for i, g in enumerate(self.filtration)}
        cornerpoint_vertices = list(set(itertools.chain.from_iterable(self.filtered_t_hubs.values())))
        self.persistence = {cornerpoint_vertex: [k for k in self.filtered_t_hubs
                                if cornerpoint_vertex in self.filtered_t_hubs[k]]
                                for cornerpoint_vertex in cornerpoint_vertices}
        self.persistence = {key: value for key, value
                                in self.persistence.iteritems()
                                if len(value) >= 1}

    @staticmethod
    def get_maximum_steady_persistence(array):
        """Return list of consecutive lists of numbers from vals (number list)."""
        sub_persistence = []
        sub_persistences = [sub_persistence]
        consecutive = None

        for value in array:
            if (value == consecutive) or (consecutive is None):
                sub_persistence.append(value)
            else:
                sub_persistence = [value]
                sub_persistences.append(sub_persistence)
            consecutive = value + 1

        sub_persistences.sort(key=len)

        return sub_persistences

    def steady_hubs_persistence(self, above_max_diagonal_gap = False,
                                gap_number = 0, epsilon = .1):
        """Ranks steady hubs according to their persistence. Recall that a
        temporary hub is steady if it lives through consecutive sublevel sets of
        the filtration induced by the weights of the graph.
        """
        self.steady_cornerpoints = []

        for vertex in self.persistence:
            pers = self.persistence[vertex]
            # birth = min(pers)
            # death = max(pers)
            # if len(pers) > 1:
            max_steady_pers = self.get_maximum_steady_persistence(pers)
            births = [min(c) for c in max_steady_pers]
            deaths = [max(c) for c in max_steady_pers]

            for birth,death in zip(births, deaths):
                if death < len(self.filtration) -1:
                    death = self.transformed_edges[death + 1]
                else:
                    death = self.transformed_edges[-1] + epsilon
                c = CornerPoint(0,
                                self.transformed_edges[birth],
                                death,
                                vertex = vertex)
                self.steady_cornerpoints.append(c)

        self.steady_pd = PersistenceDiagram(cornerpoints = self.steady_cornerpoints)
        if above_max_diagonal_gap:
            _,_ = self.steady_pd.get_nth_widest_gap(n = gap_number)
            self.gap_number = gap_number

    def ranging_hubs_persistence(self, above_max_diagonal_gap = False,
                                 gap_number = 0, epsilon = .1):
        """Ranks ranging hubs according to their persistence. Recall that a
        temporary hub is said ranging if there exist Gm and Gn sublevel sets
        (m < n) in which the temporary hub is alive.
        """
        self.ranging_cornerpoints = []

        for vertex in self.persistence:
            pers = self.persistence[vertex]
            # if len(pers) > 1:
            birth = min(pers)
            death = max(pers)
            if death < len(self.filtration)-1:
                death = self.transformed_edges[death + 1]
            else:
                death = self.transformed_edges[-1] + epsilon
            self.ranging_cornerpoints.append(CornerPoint(0,
                                                  self.transformed_edges[birth],
                                                  death,
                                                  vertex = vertex))

        self.ranging_pd = PersistenceDiagram(cornerpoints = self.ranging_cornerpoints)
        if above_max_diagonal_gap:
            _,_ = self.ranging_pd.get_nth_widest_gap(n = gap_number)
            self.gap_number = gap_number


    def get_path_weight(self, path):
        """The weight associated to a path is simply the sum of the transformed
        weight on the edges between subsequent nodes in the path"""
        return max([self.transformed_edges_dict[(v1, v2)]
                    if (v1, v2) in self.transformed_edges_dict.keys()
                    else self.transformed_edges_dict[(v2, v1)]
                    for v1,v2 in grouped(path, 2)])

    def plot_filtration(self):
        """Plots all the subgraphs of self.G given by considering the sublevel
        sets of the function defined on the weighted edges
        """
        fig, self.ax_arr = plt.subplots(int(np.ceil(len(self.filtration) / 3)),3)
        self.ax_arr = self.ax_arr.ravel()
        ordinals = ['st', 'nd', 'rd']
        for i, h in enumerate(self.filtration):
            title = str(i + 1) + ordinals[i] if i < 3 else str(i + 1) + 'th'
            self._draw(graph = h, plot_weights = True, ax = self.ax_arr[i],
                        title = title + " sublevel set")


    def _draw(self, graph = None, plot_weights = True, ax = None, title = None):
        """Plots a graph using networkx wrappers

        Parameters
        ----------
        graph : <networkx graph>
            A graph instance of networkx.Graph()
        plot_weights : bool
            If True weights are plotted on the edges of the graph
        ax : <matplotlib.axis._subplot>
            matplotlib axes handle
        title : string
            title to be attributed to ax
        """
        if graph is None:
            graph = self.G
        pos = nx.spring_layout(graph)
        if title is not None:
            ax.set_title(title)
        nx.draw(graph, pos = pos, ax = ax, with_labels = True)
        if plot_weights:
            labels = nx.get_edge_attributes(graph,'weight')
            labels = {k : np.round(v, decimals = 2) for k,v in labels.items()}
            nx.draw_networkx_edge_labels(graph, pos, edge_labels = labels,
                                        ax = ax)

    def plot_steady_persistence_diagram(self, export_json = False,
                                         return_attr = False, show = False):
        """Uses gudhi and networkx wrappers to plot the persistence diagram and
        the hubs obtained through the steady-hubs analysis, respectively
        """
        fig = plt.figure()
        self.steady_ax_arr = [plt.subplot2grid((5, 5), (0, 1), colspan = 3, rowspan = 4),
                              plt.subplot2grid((5, 5), (4, 2), colspan = 1, rowspan = 1)]
        if hasattr(self, 'steady_cornerpoints'):
            self.steady_pd.plot_gudhi(self.steady_ax_arr[1],
                    persistence_to_plot = self.steady_pd.persistence_to_plot)
            if hasattr(self.steady_pd, 'proper_cornerpoints_above_gap'):
                self.steady_pd.plot_nth_widest_gap(ax_handle = self.steady_ax_arr[1],
                                                   n = self.gap_number)
                self.steady_pd.mark_points_above_diagonal_gaps(ax_handle = self.steady_ax_arr[1])
            node_sizes_dict = {n : 0 for n in self.G.nodes}
            self.steady_pd.get_proper_cornerpoints()
            max_finite_persistence = self.steady_pd.proper_cornerpoints[-1].persistence
            finite_persistence_std = np.std([c.persistence for c in self.steady_pd.proper_cornerpoints])
            finite_persistence_min = np.min([c.persistence for c in self.steady_pd.proper_cornerpoints])
            if hasattr(self.steady_pd, 'proper_cornerpoints_above_gap'):
                for c in self.steady_cornerpoints:
                    if c.is_proper and c.above_the_gap:
                        node_sizes_dict[c.vertex] = c.persistence
                    elif not c.above_the_gap:
                        pass
                    else:
                        if finite_persistence_std == 0:
                            finite_persistence_std = 1
                        node_sizes_dict[c.vertex] = max_finite_persistence + 2 * finite_persistence_std
            else:
                for c in self.steady_cornerpoints:
                    if c.is_proper:
                        node_sizes_dict[c.vertex] = c.persistence
                    else:
                        if finite_persistence_std == 0:
                            finite_persistence_std = 1
                        node_sizes_dict[c.vertex] = max_finite_persistence + 2 * finite_persistence_std

            nodes_color_map = ['blue' if node in self.persistence.keys() else
                               'red' for node in node_sizes_dict.keys()]
            nodelist = node_sizes_dict.keys()
            node_size = [v * 50
                        if v != 0 else finite_persistence_min * 10
                        for v in node_sizes_dict.values()]
            if show:
                pos = nx.circular_layout(self.G)
                self.steady_ax_arr[0].set_title("Steady hubs persistence")
                nx.draw(self.G, pos = pos, ax = self.steady_ax_arr[0],
                        node_color = nodes_color_map,
                        nodelist = nodelist,
                        node_size = node_size,
                        with_labels = True)
                labels = nx.get_edge_attributes(self.G,'weight')
                labels = {k : np.round(v, decimals = 2) for k,v in labels.items()}
                nx.draw_networkx_edge_labels(self.G, pos, edge_labels = labels,
                                             ax = self.steady_ax_arr[0])
            if export_json is not None:
                steady_json = self.export_graph_and_hubs_as_json(nodes_color_map,
                                                                  nodelist,
                                                                  node_size,
                                                                  self.steady_pd.proper_cornerpoints)
                name = export_json + '_steady.json',
                with open(name, 'w') as outfile:
                    json.dump(steady_json, outfile)
            if return_attr:
                return nodes_color_map, nodelist, node_size, self.steady_pd.proper_cornerpoints

    def plot_ranging_persistence_diagram(self, export_json = False,
                                         return_attr = False, show = False):
        """Uses gudhi and networkx wrappers to plot the persistence diagram and
        the hubs obtained through the ranging-hubs analysis, respectively
        """
        fig = plt.figure()
        self.ranging_ax_arr = [plt.subplot2grid((5, 5), (0, 1), colspan = 3,
                                                rowspan = 4),
                              plt.subplot2grid((5, 5), (4, 2), colspan = 1,
                                               rowspan = 1)]
        if hasattr(self, 'ranging_cornerpoints'):
            self.ranging_pd.plot_gudhi(self.ranging_ax_arr[1],
                    persistence_to_plot = self.ranging_pd.persistence_to_plot)
            if hasattr(self.ranging_pd, 'proper_cornerpoints_above_gap'):
                self.ranging_pd.plot_nth_widest_gap(ax_handle = self.ranging_ax_arr[1],
                                                    n = self.gap_number)
                self.ranging_pd.mark_points_above_diagonal_gaps(ax_handle =
                                                                self.ranging_ax_arr[1])

            node_sizes_dict = {n : 0 for n in self.G.nodes}
            self.ranging_pd.get_proper_cornerpoints()
            max_finite_persistence = self.ranging_pd.proper_cornerpoints[-1].persistence
            finite_persistence_std = np.std([c.persistence for c in
                                        self.ranging_pd.proper_cornerpoints])
            finite_persistence_min = np.min([c.persistence for c in
                                        self.ranging_pd.proper_cornerpoints])
            if hasattr(self.ranging_pd, 'proper_cornerpoints_above_gap'):

                for c in self.ranging_cornerpoints:
                    if c.is_proper and c.above_the_gap:
                        node_sizes_dict[c.vertex] = c.persistence
                    elif not c.above_the_gap:
                        pass
                    else:
                        if finite_persistence_std == 0:
                            finite_persistence_std = 1
                        node_sizes_dict[c.vertex] = max_finite_persistence + 2 * finite_persistence_std
            else:

                for c in self.ranging_cornerpoints:
                    if c.is_proper:
                        node_sizes_dict[c.vertex] = c.persistence
                    else:
                        if finite_persistence_std == 0:
                            finite_persistence_std = 1
                        node_sizes_dict[c.vertex] = max_finite_persistence + 2 * finite_persistence_std
            nodes_color_map = ['blue' if n in self.persistence.keys() else 'red' for n in node_sizes_dict.keys()]
            pos = nx.circular_layout(self.G)
            self.ranging_ax_arr[0].set_title("Ranging hubs persistence")
            nodelist = node_sizes_dict.keys()
            node_size = [v * 50
                        if v != 0 else finite_persistence_min * 10
                        for v in node_sizes_dict.values()]
            if show:
                nx.draw(self.G, pos = pos, ax = self.ranging_ax_arr[0],
                        node_color = nodes_color_map,
                        nodelist = nodelist,
                        node_size = node_size,
                        with_labels = True)
                labels = nx.get_edge_attributes(self.G,'weight')
                labels = {k : np.round(v, decimals = 2) for k,v in labels.items()}
                nx.draw_networkx_edge_labels(self.G, pos, edge_labels = labels,
                                            ax = self.ranging_ax_arr[0])
            if export_json is not None:
                ranging_json = self.export_graph_and_hubs_as_json(nodes_color_map,
                                                                   nodelist,
                                                                   node_size,
                                                                   self.ranging_pd.proper_cornerpoints)
                name = export_json + '_ranging.json'
                with open(name, 'w') as outfile:
                    json.dump(ranging_json, outfile)
            if return_attr:
                return nodes_color_map, nodelist, node_size, self.ranging_pd.proper_cornerpoints


    def export_graph_and_hubs_as_json(self, nodes_color_map, nodelist,
                                       node_size, cornerpoints):
        """Export json file to allow 3d webGL visualisation

        Parameters
        ----------
        nodes_color_map : list
            list of nodes colors
        nodelist : list
            list of nodes
        node_size : list
            list of nodes sizes
        cornerpoints : list
            list of <CornerPoint> instances sorted according to their
            persistence
        """
        cornerpoints = cornerpoints[::-1]
        is_hub = ['hub' if n in self.persistence.keys() else 'not_hub'
                    for n in nodelist]
        persistence_dict = {c.vertex : c.persistence for c in cornerpoints}
        hub_rank_dict = {c.vertex : i + 1 for i,c in enumerate(cornerpoints)}
        main_attr = {'nodes': [], 'links' : []}
        for node, color, hub_or_not in zip(nodelist, nodes_color_map, is_hub):
            persistence = str(persistence_dict[node]) if hub_or_not == 'hub' else '0'
            hub_rank = str(hub_rank_dict[node]) if persistence != '0' else '0'
            main_attr['nodes'].append({'id': node,
                                       'user': hub_or_not,
                                       'description': node + ' rank ' + hub_rank + ' persistence value ' + persistence})

        for source, target in self.G.edges:
            main_attr['links'].append({'source': source, 'target': target})

        return main_attr

    def export_graph_and_hubs_as_dict(self, nodes_color_map, nodelist,
                                       node_size, cornerpoints,
                                       observation = 0):
        """Export dict to allow stream chart visualisation. The dictionary has
        structure
        g_dict = {vertex: [], persistence : [], observation = []}

        Parameters
        ----------
        nodes_color_map : list
            list of nodes colors
        nodelist : list
            list of nodes
        node_size : list
            list of nodes sizes
        cornerpoints : list
            list of <CornerPoint> instances sorted according to their
            persistence
        """
        cornerpoints = cornerpoints[::-1]
        is_hub = ['hub' if n in self.persistence.keys() else 'not_hub'
                    for n in nodelist]
        persistence_dict = {c.vertex : c.persistence for c in cornerpoints}
        hub_rank_dict = {c.vertex : i + 1 for i,c in enumerate(cornerpoints)}
        g_dict = {"vertex": [], "persistence" : [], "observation" : []}

        for node, color, hub_or_not in zip(nodelist, nodes_color_map, is_hub):
            persistence = str(persistence_dict[node]) if hub_or_not == 'hub' else '0'
            hub_rank = str(hub_rank_dict[node]) if persistence != '0' else '0'
            g_dict['vertex'].append(node)
            g_dict['persistence'].append(persistence)
            g_dict['observation'].append(observation)

        return g_dict


if __name__ == "__main__":
    def opp(array):
        return -array

    def identity(array):
        return array

    def max_(array):
        return np.max(array) - array

    steady_1 = [("a", "h", 7),
                ("b", "h", 4),
                ("c", "d", 5),
                ("d", "e", 6),
                ("f", "h", 9),
                ("g", "h", 8),
                ("h", "d", 3)]

    steady_2 = [("a", "h", 5),
                ("b", "h", 2),
                ("c", "d", 6),
                ("d", "e", 8),
                ("f", "h", 11),
                ("g", "h", 7),
                ("h", "d", 1)]

    ranging_1 = [("a", "g", 7),
                 ("b", "h", 6),
                 ("c", "h", 8),
                 ("d", "h", 5),
                 ("e", "g", 4),
                 ("f", "g", 9),
                 ("h", "g", 3)]

    ranging_2 = [("a", "g", 5),
                 ("b", "h", 7),
                 ("c", "h", 9),
                 ("d", "h", 6),
                 ("e", "g", 2),
                 ("f", "g", 8),
                 ("h", "g", 1)]

    graph = WeightedGraph(ranging_1)
    graph.build_graph()
    graph.build_filtered_subgraphs(weight_transform = identity)
    graph.get_temporary_hubs_along_filtration()
    graph.steady_hubs_persistence()
    graph.ranging_hubs_persistence()
    graph.plot_steady_persistence_diagram(show = True)
    graph.plot_ranging_persistence_diagram(show = True)
    plt.show()
