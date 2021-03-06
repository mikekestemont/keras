from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.datasets import reuters
from keras.models import Graph
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.advanced_activations import HierarchicalSoftmax
from keras.regularizers import l2
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

import time
'''
    Train and evaluate a simple MLP on the Reuters newswire topic classification task.
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python examples/reuters_mlp.py
    CPU run command:
        python examples/reuters_mlp.py
'''

max_words = 5000
batch_size = 100
nb_epoch = 1000

print("Loading data...")
(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

nb_classes = np.max(y_train)+1
print(nb_classes, 'classes')

true_labels = np.asarray([[y] for y in y_train], dtype='int8')

print("Vectorizing sequence data...")
tokenizer = Tokenizer(nb_words=max_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode="binary")
X_test = tokenizer.sequences_to_matrix(X_test, mode="binary")
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print("Convert class vector to binary class matrix (for use with categorical_crossentropy)")
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

print(true_labels.shape)

print("Building model...")
m = Graph()
m.add_input(name='input', ndim=2)
m.add_input(name='target_labels', ndim=2)


dense_output_size = 512 

# standard hidden layer:
m.add_node(Dense(max_words, dense_output_size, activation='relu'), name='dense', input='input')
#m.add_node(Dropout(0.5), name='dropout', input='dense')

# add Hierarchical Softmax:
m.add_node(HierarchicalSoftmax(input_dim=dense_output_size, output_dim=nb_classes,
                               activation='relu', W_regularizer=l2(.01),
                               train_mode='single_target', test_mode='single_target'),
           name='HierarchicalSoftmax', inputs=['dense', 'target_labels'], merge_mode='concat')

m.add_output(name='output', input='HierarchicalSoftmax')

m.compile('RMSprop', {'output': 'categorical_crossentropy'})

for i in range(250):
    history = m.fit({'input': X_train, 'target_labels': true_labels, 'output': Y_train},
                 validation_data=None, validation_split=None, nb_epoch=1,
                 batch_size=batch_size, verbose=1, shuffle=True)
    start = time.time()
    predictions = m.predict({'input': X_train, 'target_labels': true_labels},
                     batch_size = batch_size)
    end = time.time()
    print(end - start)
    predictions = np_utils.categorical_probas_to_classes(predictions['output'])
    accuracy = np_utils.accuracy(predictions, y_train)
    print(accuracy)

