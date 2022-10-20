import math
import random
import numpy as np


def vectorToDistMatrix(cities):
    '''
    Create the distance matrix
    '''
    coordinates = cities[:, :2].copy()
    cortes_penalty = cities[:, 2:3].copy()
    mexico_penalty = cities[:, 3:4].copy()
    distance_matrix = np.sqrt((np.square(coordinates[:, np.newaxis] - coordinates).sum(axis=2)))
    golf_cortes_penalty = 4 * (np.square(cortes_penalty[:, np.newaxis] - cortes_penalty).sum(axis=2))
    golf_mexico_penalty = 4 * (np.square(mexico_penalty[:, np.newaxis] - mexico_penalty).sum(axis=2))

    return distance_matrix + golf_mexico_penalty + golf_cortes_penalty


def nearestNeighbourSolution(dist_matrix):
    '''
    Computes the initial solution (nearest neighbour strategy)
    '''
    node = random.randrange(len(dist_matrix))
    result = [node]

    nodes_to_visit = list(range(len(dist_matrix)))
    nodes_to_visit.remove(node)

    while nodes_to_visit:
        nearest_node = min([(dist_matrix[node][j], j) for j in nodes_to_visit], key=lambda x: x[0])
        node = nearest_node[1]
        nodes_to_visit.remove(node)
        result.append(node)

    return result