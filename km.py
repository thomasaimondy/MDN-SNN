import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
from itertools import cycle

pn = ['a', 'b', 'c', 'd']
plt.figure(1)

task = 'MNIST' #1
# task = 'TiDigits' #2

if task == 'MNIST':
    num = '5'
elif task == 'TiDigits':
    num = '26'

rplace = '../dynamic_data/T90_0.01_0.0001/'
splace = './'

for i in range(1):
    for j in range(2):
        x = str(i)
        w = str(i + 1)
        y = str(pn[2 * j])
        z = str(pn[2 * j + 1])

        a1 = np.load(rplace + task + 'layer' + x + '_' + y + num + '.npy')
        b1 = np.load(rplace + task + 'layer' + x + '_' + z + num + '.npy')
        a2 = np.load(rplace + task + 'layer' + w + '_' + y + num + '.npy')
        b2 = np.load(rplace + task + 'layer' + w + '_' + z + num + '.npy')

        print('a1 size:', a1.shape)
        print('b1 size:', b1.shape)
        print('a2 size:', a2.shape)
        print('b2 size:', b2.shape)

        a1 = a1.reshape((a1.size, 1))
        b1 = b1.reshape((b1.size, 1))
        a2 = a2.reshape((a2.size, 1))
        b2 = b2.reshape((b2.size, 1))

        a = np.concatenate((a1, a2), axis=0)
        b = np.concatenate((b1, b2), axis=0)
        X = np.concatenate((a, b), axis=1)

        bandwidth = estimate_bandwidth(X, quantile=0.5)
        print(bandwidth)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)

        if j == 0 and task == 'TiDigits':
            n_clusters = 3

        km = KMeans(n_clusters = n_clusters)

        km.fit(X)

        labels = km.labels_
        print(labels)
        cluster_centers = km.cluster_centers_
        print('cluster_centers', cluster_centers)

        plt.clf()

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=10)
        plt.title('Estimated number of clusters: %d' % n_clusters)

        plt.savefig(splace + task + num + '_km_' + x + w + '_' + y + z)