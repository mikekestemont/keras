from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.datasets import reuters
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import HierarchicalSoftmax
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

'''
    Train and evaluate a simple MLP on the Reuters newswire topic classification task.
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python examples/reuters_mlp.py
    CPU run command:
        python examples/reuters_mlp.py
'''

max_words = 1000
batch_size = 32
nb_epoch = 5

print("Loading data...")
(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

nb_classes = np.max(y_train)+1
print(nb_classes, 'classes')

y_train = np.array(y_train, dtype='int8')

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

print("Building model...")
m = Graph()
m.add_input(name='input', ndim=2)
m.add_input(name='true_labels', ndim=1)

# standard hidden layer:
m.add_node(Dense(max_words, 512, activation='relu'), name='dense', input='input')

# add Hierarchical Softmax:
m.add_node(HierarchicalSoftmax(input_dim=512, output_dim=nb_classes),
           name='HierarchicalSoftmax', inputs=['dense', 'true_labels'],
           merge_mode='concat')

m.add_output(name='output', input='HierarchicalSoftmax')

m.compile('SGD', {'output': 'categorical_crossentropy'})

history = m.fit({'input': X_train, 'true_labels': y_train, 'output': Y_train},
                 nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)

