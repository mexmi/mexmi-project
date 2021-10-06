from base_sss import SubsetSelectionStrategy
import base_sss
import random
import copy

from sklearn.neighbors import kneighbors_graph
# from sklearn.metrics import pairwise_distances

import numpy as np

class GraphDensitySelectionStrategy(SubsetSelectionStrategy):
    def __init__(self, size, Y_vec, init_cluster, previous_s=None):
        self.previous_s = previous_s
        self.init_cluster = init_cluster
        super(GraphDensitySelectionStrategy, self).__init__(size, Y_vec)
        self.gamma = 1. / self.Y_vec.shape[1] #the second dimension
        # self.compute_graph_density()
    # its not Y_vec,but the all Y!
    def pairwise_distances(self, x,y):
        #manhaton
        w= np.expand_dims(np.sum(abs(x-y), axis=1),axis=0)
        return w
    def compute_graph_density(self, Y_e, n_neighbor=10):
        print("compute_graph_density")
        # kneighbors graph is constructed using k=10
        Y_conn = np.vstack((Y_e, self.init_cluster))
        connect = kneighbors_graph(Y_conn, n_neighbor, p=1)
        # Make connectivity matrix symmetric, if a point is a k nearest neighbor of
        # another point, make it vice versa
        neighbors = connect.nonzero()
        inds = zip(neighbors[0], neighbors[1])
        # Graph edges are weighted by applying gaussian kernel to manhattan dist.
        # By default, gamma for rbf kernel is equal to 1/n_features but may
        # get better results if gamma is tuned.
        print("start entry fro inds loop")
        for entry in inds:
            # print("pairwise is it?1")
            i = entry[0]
            j = entry[1]
            # print("pairwise is it?2")
            # print("Y_conn[[i]],", np.asarray(Y_conn[[i]]).shape)
            # print("Y_conn[[j]],", np.asarray(Y_conn[[j]]).shape)
            distance = self.pairwise_distances(np.asarray(Y_conn[[i]]), np.asarray(Y_conn[[j]]))#pairwise_distances(np.asarray(Y_conn[[i]]), np.asarray(Y_conn[[j]]), metric='manhattan')
            # print("pairwise is it?3")
            distance = distance[0, 0]
            # print("pairwise is it?4")
            weight = np.exp(-distance * self.gamma)
            # print("pairwise is it?5")
            connect[i, j] = weight
            connect[j, i] = weight

        self.connect = connect
        # Define graph density for an observation to be sum of weights for all
        # edges to the node representing the datapoint.  Normalize sum weights
        # by total number of neighbors.
        graph_density = np.zeros(Y_conn.shape[0])
        for i in np.arange(Y_conn.shape[0]):
            graph_density[i] = connect[i, :].sum() / (connect[i, :] > 0).sum()
        # self.starting_density = copy.deepcopy(self.graph_density)
        return graph_density

    def get_subset(self):
        if self.previous_s is not None:
            # print("self.previous:", np.asarray(self.previous_s).shape)
            Y_e = np.asarray([self.Y_vec[int(ie)] for ie in self.previous_s])
        else:
            Y_e = self.Y_vec
        # X = self.init_cluster #the all selected points (put into copy model again)
        # print("Y_e", np.asarray(Y_e).shape)
        # print("Y_vec", np.asarray(self.Y_vec).shape)
        #got_Y_e
        s=[]
        graph_density = self.compute_graph_density(Y_e)
        graph_density[len(Y_e):] = min(graph_density) - 1
        while len(s) < self.size:
            selected = np.argmax(graph_density)
            neighbors = (self.connect[selected, :] > 0).nonzero()[1]
            graph_density[neighbors] = graph_density[neighbors] - graph_density[selected]
            s.append(selected)
            graph_density[len(Y_e):] = min(graph_density) - 1
            graph_density[s] = min(graph_density) - 1

        if self.previous_s is not None:
            final_s=[self.previous_s[int(e)] for e in s]
        else:
            final_s=s
        return final_s