import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from gpa.utils import get_powerset_, get_supersets
from gpa.persistence_diagram import CornerPoint
from gpa.persistence_diagram import PersistenceDiagram

class PersistentEdgeBlock:
    def __init__(self, subgraph):
        self.v = set(subgraph.nodes)
        self.edges = subgraph.edges
        self.weights = [el[-1]['weight'] for el in list(self.edges.data())]
        self.birth = max(self.weights)
        self.death = max(self.weights)
        self.root = []
        self.branch = []

    def __eq__(self, other): return self.v == other.v


    def __ge__(self, other): return self.v >= other.v


    def __le__(self, other): return self.v <= other.v


    def __repr__(self):
        return "Edge block\nvertices: {}\nbirth: {}\ndeath: {}".format(self.v,
                                                                       self.birth,
                                                                       self.death)

    def is_finite(self):
        return self.death < np.inf


    def merges_in(self, other, update_vertices_to_maximal = True):
        death = (self <= other and self.birth > other.birth)
        embed = (self <= other and self.birth <= other.birth)
        merges = True
        if update_vertices_to_maximal and death:
            self.death = max(self.weights + other.weights)
            self.branch = []
        elif update_vertices_to_maximal and embed:
            self.v = other.v
            other.birth = self.birth
            other.root.append(self)
            self.death = max(self.weights + other.weights)
            self.branch = other
        else:
            merges = False
        return merges


class WeightedGraph:
    """
    Attributes
    ----------

    weighted_edges : list
        list of tuples of the form [(v1, v2, w12), ...] where v1 and v2 are the
        labels associated to the vertices on the boundary of the edge and w12
        is its weight.
    """
    def __init__(self, weighted_edges = None, nx_graph = None, is_directed=False):
        if nx_graph is not None:
            self.G = nx_graph
        elif weighted_edges is not None:
            self.weighted_edges = weighted_edges
            self.build_graph()
        else:
            raise ValueError("Specify graph either as a list of weighted edges,"
                             + " or as a networkx graph")
        self.is_directed = is_directed

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
        self.edges = list(nx.get_edge_attributes(subgraph,'weight').keys())
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
    def identity(array):
        """Standard filtrating function: do nothing
        """
        return array


    def get_filtration_values(self, subgraph, func):
        """Evaluates func on the weights defined on the edges
        """
        return func(np.asarray(list(nx.get_edge_attributes(subgraph,
                                                           'weight').values())))


    def get_subgraph_edges(self, value):
        """Returns the edges of self.G part of the sublevel set defined by value
        """
        return [edge + tuple([self.transformed_edges_dict[edge]])
                    for edge in self.transformed_edges_dict
                    if self.transformed_edges_dict[edge] <= value]


    def get_subgraph(self, edges):
        """Returns the subgraph defined by edges. Once the filtration is
        generated we are only interested in the 'hubbiness' of a
        subgraph.
        """
        H = nx.Graph() if not self.is_directed else nx.DiGraph()
        [H.add_edge(v1, v2, weight =  np.round(w12, decimals = 2))
         for v1, v2, w12 in edges]
        return H


    def build_filtered_subgraphs(self, weight_transform = None):
        """Generates the filtration of G given the values of the filtrating
        function.
        """
        if weight_transform is None:
            weight_transform = self.identity
        self.weight_transform = weight_transform
        self.get_edge_filtration_values(weight_transform = weight_transform)
        self.filtration = []
        self.transformed_edges = np.unique(self.transformed_edges)
        self.transformed_edges.sort()

        for value in self.transformed_edges:
            edges = self.get_subgraph_edges(value)
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
    
    def is_source(self, subgraph, node, weighted = True):
        in_edges = subgraph.in_edges(node)
        out_edges = subgraph.out_edges(node)
        if not weighted:
            return out_edges > in_edges
        else:
            weights = nx.get_edge_attributes(subgraph, "weight")
            in_weight = sum([weights[e] for e in in_edges])
            out_weight = sum([weights[e] for e in out_edges])
            return out_weight > in_weight
        
    def is_sink(self, subgraph, node, weighted=True):
        in_edges = subgraph.in_edges(node)
        out_edges = subgraph.out_edges(node)
        if not weighted:
            return out_edges < in_edges
        else:
            weights = nx.get_edge_attributes(subgraph, "weight")
            in_weight = sum([weights[e] for e in in_edges])
            out_weight = sum([weights[e] for e in out_edges])
            return out_weight < in_weight
    

        
    def get_vertex_feature_along_filtration(self, weighted=True, feature_name="sources"):
        if feature_name == "sources": 
            func = self.get_sources
        elif feature_name == "sinks":
            func= self.get_sinks
        else:
            raise NotImplementedError("sources and sinks are the only implemented features")
        
        self.filtered_sources = {i: func(subgraph=g, weighted=weighted)
                                 for i, g in enumerate(tqdm(self.filtration))}
        cornerpoint_vertices = self.get_list_of_unique_vertices(self.filtered_sources)
        self.cornerpoint_vertices = cornerpoint_vertices
        self.persistence = {i: [k for k in self.filtered_sources
                                if cornerpoint_vertex in self.filtered_sources[k]]
                                for i, cornerpoint_vertex in enumerate(cornerpoint_vertices)}
        self.persistence = {key: value for key, value in self.persistence.items()
                            if len(value) >= 1}
    
    def get_sources(self, subgraph=None, weighted=True):
        if subgraph is None:
            subgraph = self.G
        return [node for node in tqdm(list(subgraph.nodes))
                if (self.is_source(subgraph, node, weighted=weighted) and
                    len(subgraph[node]) > 1)]
        
    def get_sinks(self, subgraph=None, weighted=True):
        if subgraph is None:
            subgraph = self.G
        return [node for node in tqdm(list(subgraph.nodes))
                if (self.is_sink(subgraph, node, weighted=weighted) and
                len(subgraph[node]) > 1)]   

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
        
        cornerpoint_vertices = self.get_list_of_unique_vertices(self.filtered_t_hubs)
        self.cornerpoint_vertices = cornerpoint_vertices
        self.persistence = {i: [k for k in self.filtered_t_hubs
                                if cornerpoint_vertex in self.filtered_t_hubs[k]]
                                for i, cornerpoint_vertex in enumerate(cornerpoint_vertices)}
        self.persistence = {key: value for key, value in self.persistence.items()
                            if len(value) >= 1}


    def get_eulerian_filtration(self, superset = False):
        if superset:
            self.eulerians = {sublevel:
                              get_supersets(self.get_eulerian_subgraphs(subgraph))
                              for sublevel, subgraph in enumerate(self.filtration)}
        else:
            self.eulerians = {sublevel:
                              self.get_eulerian_subgraphs(subgraph)
                              for sublevel, subgraph in enumerate(self.filtration)}
        self.cornerpoint_vertices = self.get_list_of_unique_vertices(self.eulerians)

        self.persistence = {i: [k for k in self.eulerians
                            if cornerpoint_vertex in self.eulerians[k]]
                            for i, cornerpoint_vertex in
                            enumerate(self.cornerpoint_vertices)}
        self.persistence = {key: value for key, value
                                in self.persistence.items()
                                if len(value) >= 1}


    def get_edge_blocks_filtration(self, edges_to_remove=1):
        self.edges_to_remove = edges_to_remove
        self.edge_blocks={}
        self.cornerpoint_vertices = []

        for sublevel, subgraph in enumerate(self.filtration):
            self.edge_blocks[sublevel] = [set(component) for component in
                                          self.get_edge_blocks_from_filt(subgraph,
                                          edges_to_remove=edges_to_remove)]

        self.cornerpoint_vertices = self.get_list_of_unique_vertices(self.edge_blocks)
        self.persistence = {i: [k for k in self.edge_blocks
                            if cornerpoint_vertex in self.edge_blocks[k]]
                            for i, cornerpoint_vertex in
                            enumerate(self.cornerpoint_vertices)}
        self.persistence = {key: value for key, value
                                in self.persistence.items()
                                if len(value) >= 1}


    def edge_block_persistence(self, edges_to_remove=1):
        self.edges_to_remove = edges_to_remove
        self.edge_blocks_ = []

        for subgraph in self.filtration:
            ebs = self.get_edge_blocks_from_filt(subgraph,
                                                 edges_to_remove=edges_to_remove,
                                                 return_subgraphs = True)
            self.edge_blocks_.append([PersistentEdgeBlock(eb) for eb in ebs])

        for i, ebs in enumerate(self.edge_blocks_[1:]):
            prev_ebs = self.edge_blocks_[i]

            for eb in ebs:
                [p_eb.merges_in(eb) for p_eb in prev_ebs]

        self.edge_blocks_ = [eb for ebs in self.edge_blocks_ for eb in ebs
                             if eb.birth != eb.death]
        self.cornerpoints = [CornerPoint(self.edges_to_remove, e.birth, e.death,
                                         vertex =e.v)
                             for e in self.edge_blocks_
                             if isinstance(e.branch, list)]
        max_weight = max([el[-1]['weight'] for el in list(self.G.edges.data())])
        [setattr(c, "death", np.inf) for c in self.cornerpoints
         if c.death == max_weight]


    @staticmethod
    def get_list_of_unique_vertices(dict_of_vertices):
        cornerpoint_vertices_ = list(chain.from_iterable(dict_of_vertices.values()))
        cornerpoint_vertices = []

        for s in cornerpoint_vertices_:
            if s not in cornerpoint_vertices:
                cornerpoint_vertices.append(s)

        return cornerpoint_vertices


    def get_blocks_filtration(self):
        self.blocks={}

        for sublevel, subgraph in enumerate(self.filtration):
            self.blocks[sublevel] = get_supersets(self.get_blocks_from_filt(subgraph))

        cornerpoint_vertices_ = list(chain.from_iterable(self.blocks.values()))
        self.cornerpoint_vertices = []

        for s in cornerpoint_vertices_:
            if s not in self.cornerpoint_vertices:
                self.cornerpoint_vertices.append(s)

        persistence_supp = {}
        persistence_tot = {}
        count = 0

        for k in self.blocks:
            if k==0:
                for cor in self.blocks[k]:
                    persistence_supp[count] = [k, k, cor]
                    count += 1
            else:
                for cor in self.blocks[k]:
                    if cor in self.blocks[k-1]:
                        pers_temp = {}
                        
                        for i in persistence_supp:
                            b,d,cor1 = persistence_supp[i]
                            if cor1==cor:
                                pers_temp[i] = [b, k, cor]
                            else:
                                pers_temp[i] = [b, d, cor1]
                        
                        persistence_supp = pers_temp
                    else:
                        block_list = {}

                        for i in persistence_supp:
                            b,d,cor1 = persistence_supp[i]
                            if cor1 <= cor: 
                                if block_list == {}:
                                    block_list[1] = [i,b,d,cor1]
                                else:
                                    block_list[list(block_list.keys())[-1]+1] = [i,b,d,cor1]

                        if block_list == {}:
                            persistence_supp[list(persistence_supp.keys())[-1]+1] = [k,k,cor]
                        else:
                            m=[0,float('inf')]
                            pers_temp = []

                            for l in block_list:
                                i, b, _, _ = block_list[l]
                                if b < m[1]:
                                    m = [i,b]

                            for l in block_list:
                                i, b, d, cor1 = block_list[l]
                                if [i,b] == m:
                                    persistence_supp[i]=[b, k, cor]
                                else:
                                    persistence_tot[len(persistence_tot)]=[b, k, cor1]
                                    del persistence_supp[i]

                        for i in persistence_supp:
                            b, d, cor1 = persistence_supp[i]
                            persistence_supp[i] = [b,k,cor1]

        for l in persistence_supp:
            b,d,cor = persistence_supp[l]
            persistence_tot[len(persistence_tot)] = [b,d,cor]

        self.persistence = {}
        
        for l in persistence_tot:
            b,d,_ = persistence_tot[l]
            self.persistence[len(self.persistence)] = [b,d]

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

    def steady_persistence(self, above_max_diagonal_gap=False, gap_number=0):
        """Ranks steady subgraphs according to their persistence. Recall that a
        temporary subgraph is steady if it lives through consecutive sublevel sets of
        the filtration induced by the weights of the graph.
        """
        self.steady_cornerpoints = []

        for vertices in self.persistence:
            pers = self.persistence[vertices]
            max_steady_pers = self.get_maximum_steady_persistence(pers)
            births = [min(c) for c in max_steady_pers]
            deaths = [max(c) for c in max_steady_pers]
            deaths = [np.inf if d == len(self.filtration)-1
                      else self.transformed_edges[d + 1] for d in deaths]

            for birth,death in zip(births, deaths):
                c = CornerPoint(0,
                                self.transformed_edges[birth],
                                death,
                                vertex = self.cornerpoint_vertices[vertices])
                self.steady_cornerpoints.append(c)

        self.steady_pd = PersistenceDiagram(cornerpoints = self.steady_cornerpoints)
        if above_max_diagonal_gap:
            _,_ = self.steady_pd.get_nth_widest_gap(n = gap_number)
            self.gap_number = gap_number

    def ranging_persistence(self, above_max_diagonal_gap=False, gap_number=0):
        """Ranks ranging subgraphs according to their persistence. Recall that a
        temporary subgraph is said ranging if there exist Gm and Gn sublevel sets
        (m < n) in which the temporary subgraph is alive.
        """
        self.ranging_cornerpoints = []

        for vertices in self.persistence:
            pers = self.persistence[vertices]
            birth = min(pers)
            death = max(pers)
            if len(pers) == 1:
                death = birth + 1
            death = np.inf if death == len(self.filtration)-1 else self.transformed_edges[death]
            c = CornerPoint(0,
                            self.transformed_edges[birth],
                            death,
                            vertex = self.cornerpoint_vertices[vertices])
            self.ranging_cornerpoints.append(c)

        self.ranging_pd = PersistenceDiagram(cornerpoints = self.ranging_cornerpoints)
        if above_max_diagonal_gap:
            _,_ = self.ranging_pd.get_nth_widest_gap(n = gap_number)
            self.gap_number = gap_number

    @staticmethod
    def get_eulerian_subgraphs(graph):
        return [set(comb) for comb in get_powerset_(list(graph.nodes))
                if (nx.is_k_edge_connected(graph.subgraph(comb),2) & nx.is_k_edge_connected(graph.subgraph(comb),1))]

    def get_edge_blocks_from_filt(self, subgraph, edges_to_remove=1,
                                  return_subgraphs = False):
        max_blocks = sorted(map(sorted,
                            nx.k_edge_components(subgraph, k=edges_to_remove+1))
                           )
        if not return_subgraphs:
            newfilt=[]

            for k in max_blocks:
                if len(k)>1:
                    newfilt.append(set(k))

            return newfilt
        else:
            subgraphs = []

            for b in max_blocks:
                sub = subgraph.subgraph(b)
                if len(list(sub.edges)) > 0:
                    subgraphs.append(sub)

            return subgraphs

    def get_blocks_from_filt(self, subgraph):
        max_blocks = nx.k_components(subgraph)
        newfilt=[]
        if len(max_blocks)>1:
            for kel in max_blocks[2]:
                if kel not in newfilt:
                    newfilt.append(set(kel))

        for k in subgraph.edges:
            newfilt.append(set(k))

        return newfilt

    @staticmethod
    def get_edge_blocks_subgraphs(graph):
        return [set(comb) for comb in get_powerset_(list(graph.nodes))
                if (nx.is_k_edge_connected(graph.subgraph(comb),2) and
                    nx.is_k_edge_connected(graph.subgraph(comb),1))]

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

    def plot_persistence_diagram(self, title = None, coloring = None):
        """Uses gudhi and networkx wrappers to plot the persistence diagram and
        the subgraphs obtained through the steady-subgraphs analysis, respectively
        """
        pds = []
        if hasattr(self, "steady_pd"):
            pds.append(self.steady_pd)
        if hasattr(self, "ranging_pd"):
            pds.append(self.ranging_pd)
        if len(pds) == 0:
            raise ValueError("Compute first a persistence diagram.")
        _, axs = plt.subplots(1, len(pds))
        
        for ax, pd in zip(axs, pds):
            pd.plot_gudhi(ax, cornerpoints = pd.cornerpoints,
                        persistence_to_plot = pd.persistence_to_plot,
                        coloring=coloring)
        if title is not None:
            plt.suptitle(title)


if __name__ == "__main__":
    example = [('a', 'b', 14),
               ('b', 'c',  4),
               ('c', 'd',  6),
               ('d', 'e',  5),
               ('e', 'f', 12),
               ('f', 'g',  8),
               ('g', 'h',  9),
               ('i', 'f', 11),
               ('i', 'a', 13),
               ('i', 'b',  2),
               ('i', 'c',  3),
               ('c', 'f', 15),
               ('c', 'e',  7),
               ('h', 'f', 10)]

    graph = WeightedGraph(weighted_edges = example, nx_graph = None)
    graph.build_filtered_subgraphs()
    graph.get_eulerian_filtration()
    graph.steady_persistence()
    graph.plot_persistence_diagram()
