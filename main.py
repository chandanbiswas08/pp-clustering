import numpy as np
import random
import os
import sys

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering
from sklearn.preprocessing import normalize


from datahelper import process_mnist_data, read_real_data, read_hcode, representative_data_recon, training_data_gen_old
from datahelper import clustering_triplets_gen, gt_triplets_gen, get_tuple, get_sample, triplet_excluded_data
from datahelper import frac_data, get_triplets_in_batch, read_triplet_file, random_triplets_gen
from datahelper import dp_hcode, write_embedding, get_global_mean_sd
from evaluation import clustering_eval
from clusterar import do_clustering

from model_core import base_network, complete_network, train_ham_embed



BATCH_SIZE=4
mnist_csv = 'RealData/mnist/mnist.csv'


class real_mnist:
    def __init__(self):
        self.vec = []
        self.ref_class = []

class hamming_mnist:
    def __init__(self):
        self.vec = []
        self.ref_class = []


if __name__ == "__main__":
    space = 'embed'
    hdim = 64
    EMB_DIM = 512
    algo = 'AgglomerativeClustering'
    tau = 0.1
    EPOCHS = 1
    LR = 0.001
    dataset = 'mnist'
    embed_file = 'RealData/mnist/mnist_70000_10_' + str(hdim) + '_10_' + str(EMB_DIM)
    hamming_file = 'RealData/mnist/mnist_70000_10_' + str(hdim) + '_10'
    triplet_file = 'RealData/mnist/triplets_70000_10_' + str(hdim)
    real_file = 'RealData/mnist/mnist_70000_10'
    num_class = 10
    dp_numbit = 0

    if (len(sys.argv) == 1):
        hdim = 32
        EMB_DIM = 512
        algo = 'KMeans'
        tau = 0.05
        space = 'th'
        dataset = 'mnist'
        num_class = 10
    else:
        space = sys.argv[1]
        hdim = int(sys.argv[2])
        EMB_DIM = int(sys.argv[3])
        if sys.argv[4] == 'kmeans':
            algo = 'KMeans'
        if sys.argv[4] == 'hac':
            algo = 'AgglomerativeClustering'
        if sys.argv[4] == 'AffinityPropagation':
            algo = 'AffinityPropagation'
        if sys.argv[4] == 'gmm':
            algo = 'GaussianMixture'
        if sys.argv[4] == 'dbscan':
            algo = 'DBSCAN'
        tau = float(sys.argv[5])
        EPOCHS = int((sys.argv[6]))
        LR = float(sys.argv[7])
        dataset = sys.argv[8]
        num_class = int((sys.argv[9]))
        dp_numbit = int(sys.argv[10])
    if dataset == 'mnist':
        embed_file = 'RealData/mnist/mnist_vec_128_' + str(hdim) + '_10_' + str(EMB_DIM)
        hamming_file = 'RealData/mnist/mnist_vec_128_' + str(hdim) + '_10'
        triplet_file = 'RealData/mnist/triplets_mnist_vec_128_' + str(hdim)
        real_file = 'RealData/mnist/mnist_vec_128'

        # embed_file = 'RealData/mnist/mnist_70000_10_' + str(hdim) + '_10_' + str(EMB_DIM)
        # hamming_file = 'RealData/mnist/mnist_70000_10_' + str(hdim) + '_10'
        # triplet_file = 'RealData/mnist/triplets_mnist_70000_10_' + str(hdim)
        # real_file = 'RealData/mnist/mnist_70000_10'

    elif dataset == '20news':
        embed_file = 'RealData/20_news_group/20_news_group.vec_' + str(hdim) + '_10_' + str(EMB_DIM)
        hamming_file = 'RealData/20_news_group/20_news_group.vec_' + str(hdim) + '_10'
        triplet_file = 'RealData/20_news_group/triplet_20_news_group.vec_' + str(hdim)
        real_file = 'RealData/20_news_group/20_news_group.vec'
    else:
        print('Give correct dataset name.')
        exit(0)

    rmnist = real_mnist()
    rmnist.vec, rmnist.ref_class = read_real_data(real_file)
    if space != 'real':
        hmnist = hamming_mnist()
        hmnist.vec, hmnist.ref_class = read_hcode(hamming_file, hdim)
    print('Generating Triplets using %s tau=%f' % (algo, tau))
    if space != 'real' and space != 'hamming' and space != 'h2e':
        alpha = clustering_triplets_gen(tau, rmnist.vec, rmnist.ref_class, hmnist.vec, triplet_file, algo, num_class, False)

    if space != 'real':
        global_mean_pos, global_sd_pos, global_mean_neg, global_sd_neg = get_global_mean_sd(real_file, hdim)
        h = hmnist.vec

    for iter in range (1):
        if space == 'real':
            print('%s algo on real space' % algo)
            km_r_clusterar = do_clustering(dataset, rmnist.vec, rmnist.ref_class, num_class, algo=algo, triplet_file=triplet_file)
        else:
            if space == 'hamming':
                print('%s algo on Hamming space %d' % (algo,hdim))
                km_h_clusterar = do_clustering(dataset, hmnist.vec, hmnist.ref_class, num_class, algo=algo, triplet_file=triplet_file)
            else:
                if space == 'h2e':
                    print('%s algo on h2e space %d' % (algo, hdim))
                    if dataset == 'mnist':
                        hmnist.vec = representative_data_recon(dataset, space, h, algo, num_class, global_mean_pos, global_sd_pos, global_mean_neg, global_sd_neg, multibin=False)
                    else:
                        hmnist.vec = representative_data_recon(dataset, space, h, algo, num_class, global_mean_pos, global_sd_pos, global_mean_neg, global_sd_neg, multibin=True)
                    km_h_clusterar = do_clustering(dataset, hmnist.vec, hmnist.ref_class, num_class, algo=algo, triplet_file=triplet_file)
                else:
                    if space == 'h-psi':
                        print('%s algo on Triplet embedding space hdim %d h-psi embed dim %d' % (algo, hdim, EMB_DIM))
                    else:
                        if space == 'h2e-psi-noprior':
                            print('%s algo on Triplet embedding space hdim %d h2e-psi-noprior embed dim %d' % (algo, hdim, EMB_DIM))
                        else:
                            print('%s algo on Triplet embedding space hdim %d h2e-psi embed dim %d' % (algo, hdim, EMB_DIM))

                        hmnist.vec = representative_data_recon(dataset, space, h, algo, num_class, global_mean_pos, global_sd_pos, global_mean_neg, global_sd_neg, multibin=True)

                    km_embed_vec, km_embed_ref_class = train_ham_embed(dataset, hmnist, hamming_file, triplet_file, EMB_DIM, algo, BATCH_SIZE, EPOCHS, LR, tau, val_split=0.1, alpha=0.2)
                    #write_embedding(km_embed_vec, km_embed_ref_class, embed_file, EMB_DIM)
                    km_he_clusterar = do_clustering(dataset, km_embed_vec, km_embed_ref_class, num_class, algo=algo, triplet_file=triplet_file)
                    hmnist.vec = km_embed_vec
                    hmnist.ref_class = km_embed_ref_class

