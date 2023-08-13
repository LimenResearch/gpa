import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gpa.read_data_from_csv import read_graph_as_df
from gpa.weighted_graph import WeightedGraph

sns.set_palette("colorblind")
sns.set_context("paper")

def read_network(path):
    df = read_graph_as_df(path, sep=",", header=None,
                          names=["source", "target", "weight", "time"])
    return nx.from_pandas_edgelist(df, source="source", target="target",
                                    edge_attr=["weight", "time"],
                                    create_using=nx.DiGraph)

def read_gt(path):
    df = pd.read_csv(path, sep=",", header=None, names=["node", "is_trusted"])
    df["is_trusted"]=df['is_trusted'].map({1:True, -1:False})
    return df

def get_data(net_path, gt_path):
    return read_network(net_path), read_gt(gt_path)

def get_boxplot(gt, g, ax=None):
    trusted_nodes = gt[gt.is_trusted==True]["node"].values
    fraudulent_nodes = gt[gt.is_trusted==False]["node"].values
    trusted = [c for c in g.steady_cornerpoints if c.vertex in trusted_nodes]
    fraudulent = [c for c in g.steady_cornerpoints if c.vertex in fraudulent_nodes]
    per_df = pd.DataFrame.from_dict({
        "persistence": 
        [c.persistence for c in trusted]+[c.persistence for c in fraudulent], 
        "trusted":[True for _ in trusted]+[False for _ in fraudulent],
        "nodes": [c.vertex for c in trusted]+[c.vertex for c in fraudulent], 
        })
    per_df.replace([np.inf, -np.inf], 
                   np.nanmax(per_df[per_df.persistence!=np.inf].persistence) + 1,
                   inplace=True)
    sns.boxplot(data=per_df, x="trusted", y="persistence", ax=ax, showmeans=True,
                meanprops={"marker":"o",
                           "markerfacecolor":"white",
                           "markeredgecolor":"black",
                           "markersize":"10"})
    return per_df
    
    

if __name__ == '__main__':
    import os
    from gpa.constants import DATA_FOLDER
    
    def identity(array): 
        return array
    
    def opp(array):
        return -array
    
    def max_2(array):
        array = array + np.abs(np.min(array))
        return np.max(array) - array
    
       
    def max_(array):
        array = array + np.abs(np.min(array))
        return np.max(array) - array
    
    def make_positive(array):
        return array + np.min(array)
    
    
    # OTC
    net_path = os.path.join(DATA_FOLDER, "fintech", "data", "otc", "otc_network.csv")
    gt_path = os.path.join(DATA_FOLDER, "fintech", "data", "otc", "otc_gt.csv")
    # # Alpha
    # net_path = os.path.join(DATA_FOLDER, "fintech", "data", "alpha", "alpha_network.csv")
    # gt_path = os.path.join(DATA_FOLDER, "fintech", "data", "alpha", "alpha_gt.csv")
    
    graph, gt = get_data(net_path, gt_path)
    # g = WeightedGraph(nx_graph=graph, is_directed=True)
    # g.build_filtered_subgraphs(weight_transform=max_)
    # g.get_vertex_feature_along_filtration(weighted=True, feature_name="sources")
    # g.steady_persistence()
    
    h = WeightedGraph(nx_graph=graph, is_directed=True)
    h.build_filtered_subgraphs(weight_transform=max_)
    h.get_vertex_feature_along_filtration(weighted=True, feature_name="sinks")
    h.steady_persistence()
    
    # f,axs = plt.subplots(1,2)
    # per_df_g = get_boxplot(gt, g, ax=axs[0])
    f, ax = plt.subplots()
    per_df_h = get_boxplot(gt, h, ax=ax)
    