import numpy as np
import random
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering
from scipy.spatial import distance
from evaluation import clustering_eval
import os

seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

real_file = 'RealData/mnist/mnist_70000_10'
mnist_csv = 'RealData/mnist/mnist.csv'


def process_mnist_data(infile, outfile):
    cols = pd.read_csv(infile, header=None, nrows=1).columns
    mnist = pd.read_csv(infile, header=None, usecols=cols[1:]).to_numpy(dtype='float')/255
    ref_class = pd.read_csv(infile, header=None, usecols=[0]).to_numpy(dtype='int')
    fp = open(outfile, 'w')
    for i in range(len(ref_class)):
        for j in range(783):
            fp.write(str(mnist[i][j]) + ' ')
        fp.write(str(mnist[i][783]) + '\t' + str(ref_class[i][0]) + '\n')
    fp.close()

# process_mnist_data(mnist_csv, real_file)

def read_real_data(file, frac=1):
    fp = open(file, 'r')
    lines = fp.readlines()
    fp.close()
    data = []
    ref_class = []
    for line in lines:
        line = line.strip('\n').split('\t')
        v = line[0].split()
        vec = []
        for i in range(len(v)):
            vec = vec + [float(v[i])]
        data.append(vec)
        ref_class.append(int(line[1]))
    data = np.asarray(data)
    ref_class = np.asarray(ref_class)
    numdata = int(frac * len(data))
    frac_x_idx = random.sample(range(len(data)), numdata)
    data = np.asarray(data[frac_x_idx], dtype='float32')
    ref_class = ref_class[frac_x_idx]
    return data, ref_class


def read_hcode(file, hdim):
    sb_fp = open(file, 'r')
    lines = sb_fp.readlines()
    sb_fp.close()
    hcode = []
    ref_class = []
    for line in lines:
        line = line.strip('\n').split('\t')
        codes = line[0].split()
        signature = []
        for i in range(len(codes)):
            signature = signature + [int(x) for x in "{0:0=64b}".format(int(codes[i]) & 0xffffffffffffffff)[::-1]]
        if hdim < 64:
            signature = signature[0:hdim]
        hcode.append(signature)
        ref_class.append(int(line[1]))
    hcode = np.asarray(hcode, dtype='float32')
    ref_class = np.asarray(ref_class)
    # hcode = (hcode + 1) * 0.5
    # hcode = ((hcode * 2) - 1) * 0.5
    # hcode = ((hcode * 0.9) + 0.1) * 1
    return hcode, ref_class

def dp_hcode(hcode, dp_numbit=-1):
    if dp_numbit == 0:
        print('Non DP')
        return hcode
    code_len = np.shape(hcode)[1]
    if dp_numbit == -1:
        dp_numbit = code_len
    print('DP Numbit=' + str(dp_numbit))
    for i in range(len(hcode)):
        dp_bit_idx = np.random.randint(0, code_len, dp_numbit)
        for j in range(len(dp_bit_idx)):
            randnum = random.random()
            if randnum > 0.5:
                randnum1 = random.random()
                if randnum1 > 0.5:
                    hcode[i][dp_bit_idx[j]] = 1
                else:
                    hcode[i][dp_bit_idx[j]] = 0
    return hcode


def get_random_sample(data, ref_class):
    rand_idx = random.randint(0, len(ref_class))
    return data[rand_idx], ref_class[rand_idx], rand_idx

def get_sample(data, ref_class, idx):
    return data[idx], ref_class[idx], idx

def clustering_triplets_gen(fracTriplesToGen, x, ref_class, h, triplet_file, algo, num_class, evaluate=False):  # a triplet for a random ref label
    """Choose a triplet (anchor, positive, negative)
    such that anchor and positive have the same label (eqv class) and
    anchor and negative have different labels (eqv class)."""
    fp = open(triplet_file+'_'+algo, "w")
    MAX_ATTEMPTS = 30
    num_triples = 0

    #h_dim = np.shape(h)[1]
    #x = np.concatenate((x, h.reshape(-1, h_dim)), axis=-1)
    #print(np.shape(x))

    numTriplesToGen = int(fracTriplesToGen * len(x))
    frac_x_idx = random.sample(range(len(x)), numTriplesToGen)
    frac_x = np.asarray(x[frac_x_idx], dtype='float32')
    frac_ref_class = ref_class[frac_x_idx]
    # num_class = len(np.unique(np.asarray(frac_ref_class, dtype='int')))

    if algo == 'AgglomerativeClustering':
        clusterar = AgglomerativeClustering(n_clusters=num_class).fit(frac_x)
    elif algo == 'DBSCAN':
        clusterar = DBSCAN(eps=3, min_samples=5, algorithm='kd_tree', n_jobs=-1).fit(frac_x)
    elif algo == 'AffinityPropagation':
        clusterar = AffinityPropagation(damping=0.7).fit(frac_x)
    else:
        clusterar = KMeans(n_clusters=num_class, init='random', random_state=0).fit(frac_x)
    frac_clusterar_class = clusterar.labels_

    if evaluate:
        print('%s Algorithm Results on train set:' % algo)
        predRefClsCount = [np.max(frac_clusterar_class) + 1, np.max(frac_ref_class) + 1]
        clustering_eval(predRefClsCount, frac_clusterar_class, frac_ref_class)

    count = 0
    for i in range(len(frac_x_idx)):
        for k in range(1):
            idx_a = frac_x_idx[i]
            clusterar_class_a = frac_clusterar_class[i]

            idx_p = None
            idx_n = None

            j = 0
            while j < MAX_ATTEMPTS:
                rand_idx = random.randint(0, len(frac_x_idx)-1)
                idx_p = frac_x_idx[rand_idx]  # continue searching until you find a +ve sample
                clusterar_class_p = frac_clusterar_class[rand_idx]
                if clusterar_class_a == clusterar_class_p:
                    break
                else:
                    idx_p = None
                j += 1

            if idx_p == None:
                continue

            j = 0
            while j < MAX_ATTEMPTS:
                rand_idx = random.randint(0, len(frac_x_idx)-1)
                idx_n = frac_x_idx[rand_idx]
                clusterar_class_n = frac_clusterar_class[rand_idx]
                if clusterar_class_a != clusterar_class_n:
                    break
                else:
                    idx_n = None
                j += 1

            if idx_n == None:
                continue

            num_triples += 1
            fp.write('{}\t{}\t{}\n'.format(idx_a, idx_p, idx_n))

            # a_p = np.sqrt(np.sum((x[idx_a]-x[idx_p]) * (x[idx_a]-x[idx_p])))
            # a_n = np.sqrt(np.sum((x[idx_a]-x[idx_n]) * (x[idx_a]-x[idx_n])))
            # if a_p > a_n:
            #     num_triples += 1
            #     fp.write('{}\t{}\t{}\n'.format(idx_a, idx_p, idx_n))
            # else:
            #     if count < 0:
            #         num_triples += 1
            #         fp.write('{}\t{}\t{}\n'.format(idx_a, idx_p, idx_n))
            #         count += 1


    print('Generated {} triplets'.format(num_triples))
    fp.close()
    if algo == 'KMeans':
        return clusterar.inertia_/len(frac_clusterar_class)
    else:
        return 0.2


def gt_triplets_gen(fracTriplesToGen, x, ref_class, triplet_file, algo):  # a triplet for a random ref label
    """Choose a triplet (anchor, positive, negative)
    such that anchor and positive have the same label (eqv class) and
    anchor and negative have different labels (eqv class)."""
    print(triplet_file+'_'+algo)
    fp = open(triplet_file+'_'+algo, "w")
    MAX_ATTEMPTS = 30
    num_triples = 0

    numTriplesToGen = int(fracTriplesToGen * len(x))
    frac_x_idx = random.sample(range(len(x)), numTriplesToGen)
    frac_ref_class = ref_class[frac_x_idx]

    for i in range(len(frac_x_idx)):
        idx_a = frac_x_idx[i]
        class_a = frac_ref_class[i]

        idx_p = None
        idx_n = None

        j = 0
        while j < MAX_ATTEMPTS:
            rand_idx = random.randint(0, len(frac_x_idx) - 1)
            idx_p = frac_x_idx[rand_idx]  # continue searching until you find a +ve sample
            class_p = frac_ref_class[rand_idx]
            if class_a == class_p:
                break
            else:
                idx_p = None
            j += 1

        if idx_p == None:
            continue

        j = 0
        while j < MAX_ATTEMPTS:
            rand_idx = random.randint(0, len(frac_x_idx) - 1)
            idx_n = frac_x_idx[rand_idx]
            class_n = frac_ref_class[rand_idx]
            if class_a != class_n:
                break
            else:
                idx_n = None
            j += 1

        if idx_n == None:
            continue

        num_triples += 1
        fp.write('{}\t{}\t{}\n'.format(idx_a, idx_p, idx_n))
        #print(ref_class[idx_a], ref_class[idx_p], ref_class[idx_n])

    print('Generated {} triplets'.format(num_triples))
    fp.close()

def random_triplets_gen(fracTriplesToGen, x, ref_class, triplet_file, algo):  # a triplet for a random ref label
    """Choose a triplet (anchor, positive, negative)
    such that anchor and positive have the same label (eqv class) and
    anchor and negative have different labels (eqv class)."""
    fp = open(triplet_file+'_'+algo, "w")
    num_triples = 0

    numTriplesToGen = int(fracTriplesToGen * len(x))
    frac_x_idx = random.sample(range(len(x)), numTriplesToGen)

    for i in range(len(frac_x_idx)):
        idx_a = frac_x_idx[i]

        rand_idx = random.randint(0, len(frac_x_idx) - 1)
        idx_p = frac_x_idx[rand_idx]  # continue searching until you find a +ve sample

        rand_idx = random.randint(0, len(frac_x_idx) - 1)
        idx_n = frac_x_idx[rand_idx]

        num_triples += 1
        fp.write('{}\t{}\t{}\n'.format(idx_a, idx_p, idx_n))

    print('Generated {} triplets'.format(num_triples))
    fp.close()
    return 0.2

def get_tuple(x, tuple):
    a_id, p_id, n_id = tuple
    a = x[int(a_id)]
    p = x[int(p_id)]
    n = x[int(n_id)]
    return a, p, n

def read_triplet_file(triplet_file, algo):
    fp = open(triplet_file+'_'+algo, "r")
    tuples = fp.readlines()
    fp.close()
    return tuples

def triplet_excluded_data(triplet_file, data, ref_class, algo):
    triplets = read_triplet_file(triplet_file, algo)
    triplet_indices = []
    for i in range(len(triplets)):
        tuple = list(map(int, triplets[i].strip('\n').split()))
        triplet_indices.append(tuple)
    triplet_indices = np.asarray(triplet_indices, dtype='int').flatten()
    triplet_indices = np.unique(triplet_indices)

    for index in sorted(triplet_indices, reverse=True):
        del data[index]
        del ref_class[index]

    return np.asarray(data), np.asarray(ref_class)

def get_global_mean_sd_old(real_file, hdim):
    mean_fp = open(real_file+'_'+str(hdim)+'_means_10')
    sd_fp = open(real_file+'_'+str(hdim)+'_SD_10')
    means = mean_fp.readlines()
    sds = sd_fp.readlines()
    mean_fp.close()
    sd_fp.close()
    mean = np.zeros(hdim)
    sd = np.zeros(hdim)
    for i in range(hdim):
        mean_ith_dim = list(map(float, means[i].strip('\n').split()))
        sd_ith_dim = list(map(float, sds[i].strip('\n').split()))
        for j in range(len(mean_ith_dim)):
            mean[i] = mean[i] + mean_ith_dim[j]
            sd[i] = sd[i] + sd_ith_dim[j]*sd_ith_dim[j]
        sd[i] = np.sqrt(sd[i])
    return mean, sd

def get_global_mean_sd(real_file, hdim):
    mean_fp = open(real_file+'_'+str(hdim)+'_means_10')
    sd_fp = open(real_file+'_'+str(hdim)+'_SD_10')
    means = mean_fp.readlines()
    sds = sd_fp.readlines()
    mean_fp.close()
    sd_fp.close()
    num_bin = int(len(list(map(float, means[0].strip('\n').split()))) / 2)

    mean_pos = np.zeros((hdim,num_bin))
    sd_pos = np.zeros((hdim,num_bin))
    mean_neg = np.zeros((hdim,num_bin))
    sd_neg = np.zeros((hdim,num_bin))

    for i in range(hdim):
        mean_ith_dim = list(map(float, means[i].strip('\n').split()))
        sd_ith_dim = list(map(float, sds[i].strip('\n').split()))
        for j in range(2*num_bin):
            if j%2 == 0:
                mean_pos[i][int(j/2)] = mean_ith_dim[j]
                sd_pos[i][int(j/2)] = sd_ith_dim[j] * sd_ith_dim[j]
            else:
                mean_neg[i][int(j/2)] = mean_ith_dim[j]
                sd_neg[i][int(j/2)] = sd_ith_dim[j] * sd_ith_dim[j]
    return mean_pos, sd_pos, mean_neg, sd_neg

def center_cal(data, num_class, label):
    centers = np.zeros((num_class,np.shape(data)[1]))
    class_ele_num = np.zeros(num_class)
    for i in range(len(data)):
        centers[label[i]] += data[i]
        class_ele_num[label[i]] += 1
    for i in range(num_class):
        centers[i] = centers[i]/class_ele_num[i]
    return centers

def training_data_gen_old(h, algo, num_class, global_mean, global_sd):
    hdim = np.shape(h)[1]
    if algo == 'AgglomerativeClustering':
        clusterar = AgglomerativeClustering(n_clusters=num_class).fit(h)
        label = clusterar.labels_
        centers = center_cal(h, num_class, label)
    else:
        clusterar = KMeans(n_clusters=num_class, init='random', random_state=0).fit(h)
        centers = clusterar.cluster_centers_
        label = clusterar.labels_
    for i in range(len(h)):
        sample = np.random.normal(centers[label[i]] + global_mean, global_sd, hdim)
        h[i] = h[i] + sigmoid(sample)
    return h


def representative_data_recon(dataset, space, h, algo, num_class, global_mean_pos, global_sd_pos, global_mean_neg, global_sd_neg, multibin=False):
    hdim = np.shape(h)[1]
    if algo == 'AgglomerativeClustering':
        clusterar = AgglomerativeClustering(n_clusters=num_class).fit(h)
        label = clusterar.labels_
        centers = center_cal(h, num_class, label)
    else:
        clusterar = KMeans(n_clusters=num_class, init='random', random_state=0).fit(h)
        # centers = clusterar.cluster_centers_
        label = clusterar.labels_
        centers = center_cal(h, num_class, label)
    numbin = len(global_mean_pos[0])
    if multibin == False:
        mean_pos = np.zeros(hdim)
        sd_pos = np.zeros(hdim)
        mean_neg = np.zeros(hdim)
        sd_neg = np.zeros(hdim)
        for j in range(hdim):
            for k in range(numbin):
                mean_pos[j] += global_mean_pos[j][k]
                sd_pos[j] += global_sd_pos[j][k]
                mean_neg[j] += global_mean_neg[j][k]
                sd_neg[j] += global_sd_neg[j][k]
            mean_pos[j] = mean_pos[j] / numbin
            mean_neg[j] = mean_neg[j] / numbin
    for i in range(len(h)):
        global_mean = np.zeros(hdim)
        global_sd = np.zeros(hdim)
        if multibin == False:
            for j in range(hdim):
                if h[i][j] > 0.5:
                    # for k in range(numbin):
                    global_mean[j] += mean_pos[j]
                    global_sd[j] += sd_pos[j]
                else:
                    # for k in range(numbin):
                    global_mean[j] += mean_neg[j]
                    global_sd[j] += sd_neg[j]
        else:
            for j in range(hdim):
                if h[i][j] > 0.5:
                    #randbin = np.random.randint(0,numbin)
                    if dataset == 'mnist':
                        randbin = np.random.randint(0, numbin)
                    else:
                        randbin = label[i]%numbin
                    global_mean[j] = global_mean_pos[j][randbin]
                    # for k in range(numbin):
                    global_sd[j] = global_sd_pos[j][randbin]
                else:
                    #randbin = np.random.randint(0,numbin)
                    if dataset == 'mnist':
                        randbin = np.random.randint(0, numbin)
                    else:
                        randbin = label[i]%numbin
                    global_mean[j] = global_mean_neg[j][randbin]
                    # for k in range(numbin):
                    global_sd[j] = global_sd_neg[j][randbin]

        sample = np.random.normal(global_mean, global_sd, hdim)
        if space == 'h2e-psi-noprior':
            h[i] = h[i] + sample
        else:
            h[i] = centers[label[i]] * (h[i] + sample)
            #h[i] = centers[label[i]] * h[i] + (1-centers[label[i]]) * sample

    return h



def frac_data(frac, data, ref_class):
    numinstance = int(frac * len(data))
    frac_data_idx = random.sample(range(len(data)), numinstance)
    frac_data = data[frac_data_idx]
    frac_ref_class = ref_class[frac_data_idx]
    return np.asarray(frac_data), np.asarray(frac_ref_class)

def sigmoid(x): 
    return 1.0/(1 + np.exp(-x))

def get_triplets_in_batch(mnist, batch_size):
    while True:
        list_a = []
        list_p = []
        list_n = []

        # generate batch_size samples of indexes from tuples
        selected_tuples = random.sample(mnist.tuples, batch_size)

        for line in selected_tuples:
            # create numpy arrays of input data
            # and labels, from each line in the file
            a, p, n = get_tuple(mnist.vec, line.strip().split('\t'))
            list_a.append(a)
            list_p.append(p)
            list_n.append(n)

        A = np.array(list_a, dtype='float32')
        P = np.array(list_p, dtype='float32')
        N = np.array(list_n, dtype='float32')

        # A = sigmoid(A)
        # P = sigmoid(P)
        # N = sigmoid(N)


        #a = np.random.normal(0.95, 0.05, np.shape(A))
        #p = np.random.normal(0.95, 0.05, np.shape(P))
        #n = np.random.normal(0.99, 0.05, np.shape(N))

        #A = A * a
        #P = P * p
        #N = N * n

        # a "dummy" label which will come in to our identity loss
        # function below as y_true. We'll ignore it.
        label = np.ones(batch_size)
        yield [A, P, N], label


def get_embedding_old(model, x):
    x_dim = np.shape(x)[1]
    a = x
    a = a.reshape(-1, x_dim, 1)
    embedding = model.predict(a)
    return np.asarray(embedding, dtype='float32')

def write_embedding(embed_vec, ref_class, embed_file, EMB_DIM):
    outf = open(embed_file, 'w')
    for i in range(len(embed_vec)):
        for j in range(EMB_DIM-1):
            outf.write(str(embed_vec[i][j]) + ' ')
        outf.write(str(embed_vec[i][EMB_DIM-1]) + '\t' + str(ref_class[i]) + '\n')
    outf.close()


# a, ref_a, idx_a = get_sample(mnist.vec, mnist.ref_class, 50002)
# p, ref_p, idx_p = get_sample(mnist.vec, mnist.ref_class, 50004)
# n, ref_n, idx_n = get_sample(mnist.vec, mnist.ref_class, 50000)
# a = a.reshape(1, np.shape(mnist.vec)[1], 1)
# p = p.reshape(1, np.shape(mnist.vec)[1], 1)
# n = n.reshape(1, np.shape(mnist.vec)[1], 1)
# print ([ref_a, ref_p, ref_n])
# A = base_model.predict(a)
# P = base_model.predict(p)
# N = base_model.predict(n)

# dist = K.sum(K.square(A-P),axis=1)
# print(dist.eval(session=K.get_session()))

# dist = K.sum(K.square(A-N),axis=1)
# print(dist.eval(session=K.get_session()))

def distance_binning(data):
    data, ref_class = read_real_data(real_file)
    num_instance = len(data)
    max_dist = -99999
    min_dist = 99999
    for i in range(num_instance-1):
        for j in range(i+1,num_instance):
            dist = distance.euclidean(data[i], data[j])
            if dist > max_dist:
                max_dist = dist
            if dist < min_dist:
                min_dist = dist
    print(max_dist, min_dist)

# distance_binning([])



