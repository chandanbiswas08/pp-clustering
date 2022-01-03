import numpy as np
import random
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers import Input, Lambda
from keras.optimizers import Adam

from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2

import pickle

from keras import backend as K

BATCH_SIZE=50
TAU=0.5
EMB_DIM=100
LAMBDA=0.5

LR = 0.0001
EPOCHS = 5
alpha = 0.2

triplet_file = '/media/hduser/ChandanWork/Dropbox/PPML/superbit-kmeans/RealData/mnist/triplets'
real_data_file = '/media/hduser/ChandanWork/Dropbox/PPML/superbit-kmeans/RealData/mnist/mnist_70000_10'
hamming_data_file = '/media/hduser/ChandanWork/Dropbox/PPML/superbit-kmeans/RealData/mnist/mnist_70000_10_128_10'

random.seed(12345)

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)


def triplet_loss(x, alpha = 0.2):
    # Triplet Loss function.
    anchor, positive, negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    return loss

class mnist:

    def __init__(self):
        self.maxlen = 784
        self.embedding_dim = EMB_DIM
        self.tuples = []

    def read_real_data(self, file):
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
        return np.asarray(data), np.asarray(ref_class)

    def read_hcode(self, file):
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
            hcode.append(signature)
            ref_class.append(int(line[1]))
        return np.asarray(hcode), np.asarray(ref_class)

    def get_random_sample(self, data):
        rand_idx = random.randint(0,len(self.ref_class))
        return data[rand_idx], self.ref_class[rand_idx], rand_idx

    def gen_and_save_triplets(self, numTriplesToGen, outFileName):  # a triplet for a random ref label
        """Choose a triplet (anchor, positive, negative)
        such that anchor and positive have the same label (eqv class) and
        anchor and negative have different labels (eqv class)."""
        fp = open(outFileName, "w")
        MAX_ATTEMPTS = 30
        num_triples = 0

        self.hcode, self.ref_class = self.read_hcode(hamming_data_file)
        self.x, _ = self.read_real_data(real_data_file)

        for i in range(numTriplesToGen):
            a, ref_a, idx_a = self.get_random_sample(self.hcode)
            idx_p = None
            idx_n = None

            j = 0
            while j < MAX_ATTEMPTS:
                p, ref_p, idx_p = self.get_random_sample(self.hcode)  # continue searching until you find a +ve sample
                if ref_a == ref_p:
                    break
                else:
                    idx_p = None
                j += 1

            if idx_p == None:
                continue

            j = 0
            while j < MAX_ATTEMPTS:
                n, ref_n, idx_n = self.get_random_sample(self.hcode)
                if ref_a != ref_n:
                    break
                else:
                    idx_n = None
                j += 1

            if idx_n == None:
                continue

            num_triples += 1
            fp.write('{}\t{}\t{}\n'.format(idx_a, idx_p, idx_n))

        print ('Generated {} triplets'.format(num_triples))
        fp.close()

    def get_tuple(self, line):
        a_id, p_id, n_id = line.strip().split('\t')
        a = self.x[int(a_id)]
        p = self.x[int(p_id)]
        n = self.x[int(n_id)]
        return a, p, n


mnst = mnist()
# mnst.gen_and_save_triplets(3000, triplet_file)
mnst.x, mnst.ref_class = mnst.read_real_data(real_data_file)
mnst.x = mnst.x.reshape(-1,mnst.maxlen,1)

fp = open(triplet_file, "r")
mnst.tuples = fp.readlines()
fp.close()
#
# with open('mnist_obj.pkl', 'wb') as mnist_obj:
#     pickle.dump(mnst, mnist_obj, pickle.HIGHEST_PROTOCOL)

# with open('mnist_obj.pkl', 'rb') as input:
#     mnst = pickle.load(input)

def get_triplets_in_batch(mnst, batch_size):
    while True:
        list_a = []
        list_p = []
        list_n = []

        # generate batch_size samples of indexes from tuples
        selected_tuples = random.sample(mnst.tuples, batch_size)

        for line in selected_tuples:
            # create numpy arrays of input data
            # and labels, from each line in the file
            a, p, n = mnst.get_tuple(line)
            list_a.append(a)
            list_p.append(p)
            list_n.append(n)

        A = np.array(list_a, dtype='float32')
        P = np.array(list_p, dtype='float32')
        N = np.array(list_n, dtype='float32')

        # a "dummy" label which will come in to our identity loss
        # function below as y_true. We'll ignore it.
        label = np.ones(batch_size)
        yield [A, P, N], label


def create_base_network(in_dims):
    model = Sequential()
    model.add(Dense(100, input_shape=(in_dims[0],in_dims[1],)))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    return model


def complete_model(base_model, mnst):

    # Create the complete model with three
    # embedding models and minimize the loss
    # between their output embeddings
    input_1 = Input((mnst.maxlen,1, ), name='anchor_input')
    input_2 = Input((mnst.maxlen,1, ), name='positive_input')
    input_3 = Input((mnst.maxlen,1, ), name='negative_input')

    A = base_model(input_1)
    P = base_model(input_2)
    N = base_model(input_3)

    triplet_indicator = Lambda(triplet_loss)([A, P, N])
    # merged_vector = concatenate([A, P, N], axis=-1, name='merged_layer')
    model = Model(inputs=[input_1, input_2, input_3], outputs=triplet_indicator)
    model.compile(loss=identity_loss, optimizer=Adam(LR))
    return model


base_model = create_base_network([mnst.maxlen,1,])
base_model.summary()
model = complete_model(base_model, mnst)
model.summary()


train_generator = get_triplets_in_batch(mnst, BATCH_SIZE)

history = model.fit_generator(train_generator,
                    epochs=2,
                    verbose=2, steps_per_epoch=20)

