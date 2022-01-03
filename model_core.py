import numpy as np
import tensorflow as tf
import os
import math
import copy

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

seed_value= 0

os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
tf.compat.v1.set_random_seed(seed_value)
#Configure a new global `tensorflow` session
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.per_process_gpu_memory_fraction=0.8
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)



from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, AveragePooling1D
from keras.optimizers import Adam, SGD
from keras.layers import Input, Conv1D, Lambda, Dense, Flatten, MaxPooling1D, concatenate, Activation
from keras.models import Model, Sequential, load_model, save_model
from keras.regularizers import l2
from keras import backend as K
from keras import activations
from keras.callbacks import ModelCheckpoint

from datahelper import get_triplets_in_batch, read_triplet_file


K.set_session(sess)


# LR = 0.001

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def my_loss_fn(y_true, y_pred):
    squared_difference = K.square(y_true - y_pred)
    return K.mean(squared_difference, axis=-1)  # Note the `axis=-1`


def triplet_loss(x, alpha = 0.2):
    # print('triplet_loss alpha ' + str(alpha))
    # Triplet Loss function.
    anchor, positive, negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    # pos_neg_dist = K.sum(K.square(positive-negative),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    # basic_loss = (pos_dist - neg_dist) / pos_neg_dist
    # basic_loss = pos_dist-(neg_dist+pos_neg_dist)+alpha
    # basic_loss = K.sigmoid(basic_loss)
    # basic_loss = K.relu(basic_loss)
    loss = K.maximum(basic_loss,0.0)
    return loss

# def base_network_old(in_dims, EMB_DIM):
#     """
#     Base network to be shared.
#     """
#     model = Sequential()
#     model.add(Conv1D(128, 7, padding='same', input_shape=(in_dims[0], in_dims[1],), activation='relu', name='conv1'))
#     model.add(MaxPooling1D(3, 3, padding='same', name='pool1'))
#     model.add(Conv1D(256, 5, padding='same', activation='relu', name='conv2'))
#     model.add(MaxPooling1D(3, 3, padding='same', name='pool2'))
#     model.add(Dropout(0.2))
#     model.add(Flatten(name='flatten'))
#     model.add(Dense(EMB_DIM, name='embeddings'))
#     return model

def base_network_best_20news(in_dims, EMB_DIM):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Conv1D(256, 4, padding='valid', input_shape=(in_dims[0], in_dims[1],), activation='tanh', name='conv1'))
    model.add(MaxPooling1D(2, 1, padding='valid', name='pool1'))
    model.add(Conv1D(512, 8, padding='valid', activation='tanh', name='conv2'))
    model.add(MaxPooling1D(2, 1, padding='valid', name='pool2'))
    model.add(Dropout(0.2))
    model.add(Flatten(name='flatten'))
    model.add(Dense(EMB_DIM, name='embeddings', kernel_regularizer='l2'))
    return model

def base_network(in_dims, EMB_DIM):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Conv1D(64, 8, padding='valid', input_shape=(in_dims[0], in_dims[1],), activation='tanh', name='conv1'))
    model.add(MaxPooling1D(3, 2, padding='valid', name='pool1'))
    model.add(Conv1D(32, 4, padding='valid', activation='tanh', name='conv2'))
    model.add(MaxPooling1D(3, 2, padding='valid', name='pool2'))
    model.add(Dropout(0.2))
    model.add(Flatten(name='flatten'))
    model.add(Dense(EMB_DIM, name='embeddings', kernel_regularizer='l2'))
    return model

def base_network_best_mnist(in_dims, EMB_DIM):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Conv1D(64, 8, padding='valid', input_shape=(in_dims[0], in_dims[1],), activation='tanh', name='conv1'))
    model.add(MaxPooling1D(3, 2, padding='valid', name='pool1'))
    model.add(Conv1D(32, 4, padding='valid', activation='tanh', name='conv2'))
    model.add(MaxPooling1D(3, 2, padding='valid', name='pool2'))
    model.add(Dropout(0.2))
    model.add(Flatten(name='flatten'))
    model.add(Dense(EMB_DIM, name='embeddings', kernel_regularizer='l2'))
    return model

def base_network_cnn(in_dims, EMB_DIM):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Conv1D(64, 8, padding='valid', input_shape=(in_dims[0], in_dims[1],), activation='tanh', name='conv1'))
    model.add(MaxPooling1D(3, 2, padding='valid', name='pool1'))
    model.add(Conv1D(128, 4, padding='valid', activation='tanh', name='conv2'))
    model.add(MaxPooling1D(3, 2, padding='valid', name='pool2'))

    model.add(Dropout(0.2))
    model.add(Flatten(name='flatten'))
    model.add(Dense(EMB_DIM, name='embeddings'))
    return model

def base_network_dense(in_dims, EMB_DIM):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Flatten(input_shape=(in_dims[0],in_dims[1],)))
    model.add(Dense(512, kernel_initializer="uniform", name='embeddings1'))
    model.add(BatchNormalization())
    model.add(Dense(256, name='embeddings2'))
    model.add(BatchNormalization())

    model.add(Dropout(0.2))
    model.add(Dense(EMB_DIM, name='embeddings3'))
    return model

def base_network_1(in_dims, EMB_DIM):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Dense(128, init='uniform', input_shape=(in_dims[0],in_dims[1],)))
    model.add(BatchNormalization())
    #model.add(Conv1D(128,7,padding='same',input_shape=(in_dims[0],in_dims[1],),activation='relu',name='conv1'))
    model.add(Conv1D(128,5,padding='same',activation='relu',name='conv1'))
    #model.add(MaxPooling1D(3,3,padding='same',name='pool1'))
    ##
    #model.add(Dense(128, init='uniform'))
    #model.add(BatchNormalization())
    ##

    model.add(Conv1D(256,5,padding='same',activation='relu',name='conv2'))
    # model.add(MaxPooling1D(3,3,padding='same',name='pool2'))
    ##
    # model.add(Dense(256, init='uniform'))
    # model.add(BatchNormalization())
    # model.add(Conv1D(512, 5, padding='same', activation='relu', name='conv3'))
    ##
    #model.add(Dropout(0.2))
    model.add(Flatten(name='flatten'))
    model.add(Dense(EMB_DIM,name='embeddings'))
    return model

# def create_base_network_lstm(in_dims):
#     model = Sequential()
#     model.add(Dense(512, input_shape=(in_dims[0],in_dims[1],)))
#     model.add(LSTM(64))
#     model.add(Dropout(0.2))
#     model.add(Dense(256, activation='relu'))
#     return model



def complete_network(base_model, data, LR=0.001, alpha=0.2):

    # Create the complete model with three
    # embedding models and minimize the loss
    # between their output embeddings
    input_1 = Input((np.shape(data.vec)[1],1, ), name='anchor_input')
    input_2 = Input((np.shape(data.vec)[1],1, ), name='positive_input')
    input_3 = Input((np.shape(data.vec)[1],1, ), name='negative_input')

    A = base_model(input_1)
    P = base_model(input_2)
    N = base_model(input_3)

    triplet_indicator = Lambda(triplet_loss, arguments={'alpha':alpha})([A, P, N])
    # triplet_indicator = concatenate([A, P, N], axis=-1, name='merged_layer')
    model = Model(inputs=[input_1, input_2, input_3], outputs=triplet_indicator)
    model.compile(loss=identity_loss, optimizer=SGD(LR))
    return model

# checkpoint_filepath = 'weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint_filepath = 'weights/best.hdf5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

def train_ham_embed(dbname, data, hamming_file, triplet_file, EMB_DIM, algo, BATCH_SIZE, EPOCHS, LR, tau, val_split=0.1, alpha=0.2):
    data.tuples = read_triplet_file(triplet_file, algo)
    data.vec = data.vec.reshape(-1,np.shape(data.vec)[1],1)
    if dbname == 'mnist':
        base_model = base_network_best_mnist([np.shape(data.vec)[1],1,], EMB_DIM)
    elif dbname == '20news':
        base_model = base_network_best_20news([np.shape(data.vec)[1], 1, ], EMB_DIM)
    else:
        base_model = base_network([np.shape(data.vec)[1], 1, ], EMB_DIM)
    base_model.summary()
    model = complete_network(base_model, data, LR, alpha)
    model.summary()
    train_set = copy.deepcopy(data)
    train_set.tuples = data.tuples[0:int((1-val_split)*len(data.tuples))]

    val_set = copy.deepcopy(data)
    val_set.tuples = data.tuples[int((1-val_split) * len(data.tuples)):len(data.tuples)]
    train_generator = get_triplets_in_batch(train_set, BATCH_SIZE)
    val_generator = get_triplets_in_batch(val_set, BATCH_SIZE)
    history = model.fit_generator(train_generator,
                                  epochs=EPOCHS,
                                  verbose=2,
                                  callbacks=[model_checkpoint_callback],
                                  validation_data=val_generator,
                                  validation_steps=20,
                                  steps_per_epoch=20)
    embed_vec = get_embedding(dbname, checkpoint_filepath, data, EMB_DIM)
    return embed_vec, data.ref_class


def get_embedding(dbname, checkpoint_filepath, x, EMB_DIM):
    if dbname == 'mnist':
        base_model = base_network_best_mnist([np.shape(x.vec)[1], 1, ], EMB_DIM)
    elif dbname == '20news':
        base_model = base_network_best_20news([np.shape(x.vec)[1], 1, ], EMB_DIM)
    else:
        base_model = base_network([np.shape(x.vec)[1], 1, ], EMB_DIM)
    model = complete_network(base_model, x)
    model.load_weights(checkpoint_filepath)
    x_dim = np.shape(x.vec)[1]
    a = x.vec
    a = a.reshape(-1, x_dim, 1)
    embedding = base_model.predict(a)
    embedding = np.asarray(embedding, dtype='float32')
    return embedding
