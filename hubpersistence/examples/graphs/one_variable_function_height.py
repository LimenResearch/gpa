from __future__ import absolute_import, print_function, division
import matplotlib
matplotlib.use("TkAgg")
import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import argrelmin, argrelmax
from hubpersistence.persistence_diagram import CornerPoint, PersistenceDiagram
sns.set()

class FunctionGraph(object):
    """Gets a list of points and
    a function in one variable and build the associated SimplexTree.

    Attributes
    ----------
    X : list
        List of x values to be fed to the function
    f : function or tuple of functions and intervals or numpy array
        Function to be applied to the points in X


    """
    def __init__(self, X, f, add_noise = False):
        self.X = sorted(X)
        self.f = f
        self.Y = self.apply_function()
        if add_noise:
            self.add_noise()


    def apply_function(self):
        if hasattr(self.f, '__call__'):
            #function
            return [self.f(x) for x in self.X]
        elif isinstance(self.f, list):
            #piecewise function
            values = []

            for f_, s, e in self.f:
                values.extend([f_(x) for x in self.X if x >= s and x < e])

            return values
        elif isinstance(self.f, np.ndarray):
            #set of values
            if len(self.f) != len(self.X):
                raise ValueError("The number of points and values does not match")
            return self.f

    def init_and_build_simplex_tree(self):
        self.st = gd.SimplexTree()

        for i in range(len(self.X) - 1):
            self.st.insert([i], filtration = self.Y[i])
            self.st.insert([i, i + 1], filtration = max(self.Y[i], self.Y[i + 1]))

        self.st.insert([i + 1], filtration = self.Y[i + 1])
        self.pd = PersistenceDiagram(self.st)

    def init_figures(self):
        fig, self.ax_arr = plt.subplots(2)
        self.plot_function(self.ax_arr[0])
        self.pd.plot_gudhi(self.ax_arr[1])

    def plot_function(self, ax):
        ax.plot(self.X, self.Y)

    def print_info_complex(self):
        result_str = 'num_vertices=' + repr(self.st.num_vertices())
        print(result_str)
        result_str = 'num_simplices=' + repr(self.st.num_simplices())
        print(result_str)

    def get_boarder_extremes(self):
        if self.Y[0] > self.Y[1]:
            self.index_maxima = np.concatenate([self.index_maxima, [0]])
        else:
            self.index_minima = np.concatenate([self.index_minima, [0]])
        if self.Y[-1] > self.Y[-2]:
            self.index_maxima = np.concatenate([self.index_maxima, [len(self.Y) - 1]])
        else:
            self.index_minima = np.concatenate([self.index_minima, [len(self.Y) - 1]])

    def mark_maxima_and_minima(self):
        self.X = np.asarray(self.X)
        self.Y = np.asarray(self.Y)
        self.index_maxima = argrelmax(self.Y)[0]
        self.index_minima = argrelmin(self.Y)[0]
        self.get_boarder_extremes()
        self.value_maxima = np.asarray([self.Y[max_ind] for max_ind in self.index_maxima])
        #mark cornerpoints in persistence diagram that correspond to a maximum with
        #a unique color
        [self.ax_arr[1].plot(cornerpoint.birth,
                            cornerpoint.death,
                            '.', markersize = 20 * self.pd.cornerpoints_multiset[cornerpoint],
                            color = self.pd.colors[i])
                            for i, cornerpoint in enumerate(self.pd.cornerpoints_multiset)
                            if len(np.where(self.value_maxima == cornerpoint.death)[0]) > 0]
        self.cornerpoint_deaths = [c.death for c in self.pd.cornerpoints_multiset]
        #mark on function
        [self.ax_arr[0].plot(self.X[ind], self.Y[ind],
                            '.', markersize = 20,
                            color = self.pd.colors[self.cornerpoint_deaths.index(self.Y[ind])])
                            for ind in self.index_maxima
                            if self.Y[ind] in self.cornerpoint_deaths]

        self.ax_arr[0].plot(self.X[self.index_minima], self.Y[self.index_minima], '.')

    def traceback_classes_above_widest_gap(self):
        c_above_gap_deaths = [c.death for c in self.pd.proper_cornerpoints_above_gap]
        [self.ax_arr[0].plot(self.X[ind], self.Y[ind],
                            'o', ms=14, markerfacecolor="None",
                            markeredgecolor='red', markeredgewidth=5,
                            color = self.pd.colors[self.cornerpoint_deaths.index(self.Y[ind])])
                            for ind in self.index_maxima
                            if self.Y[ind] in c_above_gap_deaths]

    def add_noise(self):
        self.Y = np.asarray(self.Y)
        noise = np.random.normal(0, .0005, self.Y.shape)
        self.Y = self.Y + noise


if __name__ == "__main__":
    import numpy as np

    plt.ion()


    def square(x):
        return - np.random.randint(1,10) * x**2

    def identity(x):
        return x

    def top_sin(x):
        return np.sin(1/x)

    # X = np.linspace(0, 10, num = 500)
    # f = FunctionGraph(X, [[identity, 0, 0.2], [top_sin, 0.2, 5], [np.cos, 5, 10.1]], add_noise = False)
    # f = FunctionGraph(X, top_sin)
    # f = FunctionGraph(X, [[square, 0, 0.2], [square, 0.2, 5], [square, 5, 10.1]], add_noise = False)
    ####
    #international airline passengers
    # Y = np.genfromtxt('../../data/international-airline-passengers.csv', delimiter=';')[1:,1]
    # X = range(len(Y))
    # Nile flow
    Y = np.genfromtxt('../../data/kurlin_height/Nile.csv', delimiter=',')[1:,2]
    X = range(len(Y))
    f = FunctionGraph(X,Y)
    ####
    f.init_and_build_simplex_tree()
    f.init_figures()
    f.pd.get_persistence_diagram()
    f.mark_maxima_and_minima()
    f.pd.plot_nth_widest_gap(f.ax_arr[1])
    f.pd.mark_points_above_diagonal_gaps(f.ax_arr[1])
    f.traceback_classes_above_widest_gap()
