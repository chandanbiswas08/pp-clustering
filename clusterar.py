from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
from evaluation import clustering_eval
from datahelper import triplet_excluded_data
from sklearn.preprocessing import normalize
from write_gt_pred import write_gt_pred

import os
import random
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

def do_clustering(dataset, data, ref_class, num_class, algo, triplet_file):
    # data = normalize(data)
    # if dataset != 'flame' and dataset != 'jain' and dataset != 'spiral':
    #     data, ref_class = triplet_excluded_data(triplet_file, data.tolist(), ref_class.tolist(), algo)
    print('Clustering algorithm name ' + algo)
    if algo == 'AgglomerativeClustering':
        clusterar = AgglomerativeClustering(n_clusters=num_class).fit(data)
    elif algo == 'DBSCAN':
        clusterar = DBSCAN(eps=5.5, min_samples=5, algorithm='kd_tree', n_jobs=-1).fit(data)
    elif algo == 'AffinityPropagation':
        clusterar = AffinityPropagation(damping=0.7).fit(data)
    elif algo == 'GaussianMixture':
        clusterar = GaussianMixture(n_components=num_class, random_state=0).fit(data)
    else:
        clusterar = KMeans(n_clusters=num_class, init='random', random_state=0, max_iter=10).fit(data)
    if algo == 'GaussianMixture':
        pred_class = clusterar.predict(data)
    else:
        pred_class = clusterar.labels_

    if algo == 'DBSCAN':
        predRefClsCount = [np.max(pred_class) + 2, np.max(ref_class) + 1]
    else:
        predRefClsCount = [np.max(pred_class) + 1, np.max(ref_class) + 1]
    clustering_eval(predRefClsCount, pred_class, ref_class)
    clusterar_out = open('pred_class', 'w')
    for i in range(len(pred_class)):
        clusterar_out.write(str(pred_class[i])+'\n')
    clusterar_out.close()
    # write_gt_pred(dataset, ref_class, pred_class)
    return clusterar
