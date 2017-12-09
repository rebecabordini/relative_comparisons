# -*- coding:utf-8 -*-
#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: relative_comparisons.py
# Version 0.0.1
# ---------------------------------------------------------------------------
#
# relative_comparisons.py -  Solves a Distance Metric from Relative Comparisons
#
# To run from the command line, use
#
# python relative_comparisons.py

from __future__ import print_function

import cplex
import numpy as np

def coordenadas_from_svd():
    d = \
    [[0, 400, 851, 1551, 1769, 1605, 2596, 1137, 1255, 1123, 188, 1282, 271, 2300, 483, 1038, 2099, 2699, 2493,393],
     [400, 0, 454, 1198, 1370, 1286, 2198, 803, 1181, 731, 292, 883, 279, 1906, 178, 662, 1699, 2300, 2117,292],
     [851, 454, 0, 803, 920, 940, 1745, 482, 1188, 355, 713, 432, 666, 1453, 410, 262, 1260, 1858, 1737, 597],
     [1551, 1198, 803, 0, 663, 225, 1240, 420, 1111, 862, 1374, 586, 1299, 887, 1070, 547, 999, 1483, 1681,1185],
     [1769, 1370, 920, 663, 0, 879, 831, 879, 1726, 700, 1631, 488, 1579, 586, 1320, 796, 371, 949, 1021, 1494],
     [1605, 1286, 940, 225, 879, 0, 1374, 484, 968, 1056, 1420, 794, 1341, 1017, 1137, 679, 1200, 1645, 1891,1220],
     [2596, 2198, 1745, 1240, 831, 1374, 0, 1603, 2339, 1524, 2451, 1315, 2394, 357, 2136, 1589, 579, 347, 959,2300],
     [1137, 803, 482, 420, 879, 484, 1603, 0, 872, 699, 957, 529, 881, 1263, 660, 240, 1250, 1802, 1867, 765],
     [1255, 1181, 1188, 1111, 1726, 968, 2339, 872, 0, 1511, 1092, 1397, 1019, 1982, 1010, 1061, 2089, 2594,2734, 923],
     [1123, 731, 355, 862, 700, 1056, 1524, 699, 1511, 0, 1018, 290, 985, 1280, 743, 466, 987, 1584, 1395, 934],
     [188, 292, 713, 1374, 1631, 1420, 2451, 957, 1092, 1018, 0, 1144, 83, 2145, 317, 875, 1972, 2571, 2408,230],
     [1282, 883, 432, 586, 488, 794, 1315, 529, 1397, 290, 1144, 0, 1094, 1036, 836, 354, 833, 1429, 1369,1014],
     [271, 279, 666, 1299, 1579, 1341, 2394, 881, 1019, 985, 83, 1094, 0, 2083, 259, 811, 1925, 2523, 2380,123],
     [2300, 1906, 1453, 887, 586, 1017, 357, 1263, 1982, 1280, 2145, 1036, 2083, 0, 1828, 1272, 504, 653, 1114,1973],
     [483, 178, 410, 1070, 1320, 1137, 2136, 660, 1010, 743, 317, 836, 259, 1828, 0, 559, 1668, 2264, 2138,192],
     [1038, 662, 262, 547, 796, 679, 1589, 240, 1061, 466, 875, 354, 811, 1272, 559, 0, 1162, 1744, 1724, 712],
     [2099, 1699, 1260, 999, 371, 1200, 579, 1250, 2089, 987, 1972, 833, 1925, 504, 1668, 1162, 0, 600, 701,1848],
     [2699, 2300, 1858, 1483, 949, 1645, 347, 1802, 2594, 1584, 2571, 1429, 2523, 653, 2264, 1744, 600, 0, 678,2442],
     [2493, 2117, 1737, 1681, 1021, 1891, 959, 1867, 2734, 1395, 2408, 1369, 2380, 1114, 2138, 1724, 701, 678,0, 2329],
     [393, 292, 597, 1185, 1494, 1220, 2300, 765, 923, 934, 230, 1014, 123, 1973, 192, 712, 1848, 2442, 2329,0]]

    cities = ['Boston', 'Buffalo', 'Chicago', 'Dallas', 'Denver', 'Houston', 'Los Angeles', 'Memphis', 'Miami',
              'Minneapolis', 'New York', 'Omaha', 'Philadelphia', 'Phoenix', 'Pittsburgh', 'Saint Louis',
              'Salt Lake City', 'San Francisco', 'Seattle', 'Washington D.C']

    XXt = np.zeros(shape=(20, 20))
    total = 0
    for i in range(len(d)):
        for j in range(len(d[i])):
            total += d[i][j]

    for i in range(len(d)):
        for j in range(len(d[i])):
            first_value = d[i][j]**2
            average_1 = sum([element**2 for element in d[i][:]])
            average_2 = sum([element**2 for element in d[:][j]])
            XXt[i][j] = (first_value - (0.05 * average_1) - (0.05 * average_2) + (0.0025 * total)) * (-0.5)

    np.set_printoptions(suppress=True)
    U, sigma, Ut = np.linalg.svd(XXt, full_matrices=True)
    #  Não pego a primeira coluna porque é a coordenada em R3
    vetores_singulares = U[:, 1:3]
    valores_singulares = np.diag(sigma)[1:3, 1:3]
    coordenadas = np.matmul(vetores_singulares, valores_singulares)
    coordenadas = coordenadas * -1.10
    latitude = coordenadas[:, 0]
    longitude = coordenadas[:, 1]
    return latitude, longitude


def relative_comparisons():
    latitude, longitude = coordenadas_from_svd()
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    relative_comparisons()