from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import theano
import six.moves.cPickle
import os, re, json

from keras.preprocessing import sequence, text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.layers.embeddings import Embedding
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.regularizers import l2
from keras.layers.advanced_activations import HierarchicalSoftmax
from keras.utils import np_utils, generic_utils

from six.moves import range
from six.moves import zip

max_features = 100000  # vocabulary size: top 50,000 most common words in data
nb_hidden = 256 # embedding space dimension

save = True
load_model = False
load_tokenizer = True
train_model = True

output_dir = os.path.expanduser("~/.keras/models/")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

model_name = "HN_skipgram_model.pkl"
tokenizer_fname = "HN_tokenizer.pkl"

data_path = os.path.expanduser("~/HNCommentsAll.1perline.json")

# text preprocessing utils
html_tags = re.compile(r'<.*?>')
to_replace = [('&#x27;', "'")]
hex_tags = re.compile(r'&.*?;')

def clean_comment(comment):
    c = str(comment.encode("utf-8"))
    c = html_tags.sub(' ', c)
    for tag, char in to_replace:
        c = c.replace(tag, char)
    c = hex_tags.sub(' ', c)
    return c

def text_generator(path=data_path):
    f = open(path)
    for i, l in enumerate(f):
        comment_data = json.loads(l)
        comment_text = comment_data["comment_text"]
        comment_text = clean_comment(comment_text)
        if i % 10000 == 0:
            print(i)
        if i >= 200000:
            break
        yield comment_text
    f.close()

# model management
if load_tokenizer:
    print('Load tokenizer...')
    tokenizer = six.moves.cPickle.load(open(os.path.join(output_dir, tokenizer_fname), 'rb'))
else:
    print("Fit tokenizer...")
    tokenizer = text.Tokenizer(nb_words=max_features)
    tokenizer.fit_on_texts(text_generator())
    if save:
        print("Save tokenizer...")
        six.moves.cPickle.dump(tokenizer, open(os.path.join(output_dir, tokenizer_fname), "wb"))

print("Compiling model...")


m = Graph()
m.add_input(name='input', ndim=2)

# plain softmax:
#m.add_node(Dense(max_features, nb_hidden), name='dense1', input='input')
#m.add_node(Dense(nb_hidden, max_features), name='dense2', input='dense1')
#m.add_node(Activation('softmax'), name='softmax', input='dense2')

# hierarchical variant:
m.add_node(Dense(max_features, nb_hidden), name='dense', input='input')
m.add_input(name='target_labels', ndim=2)
m.add_node(HierarchicalSoftmax(input_dim=nb_hidden, output_dim=max_features,
                               activation='relu', W_regularizer=l2(.01),
                               train_mode='single_target', test_mode='all_targets'),
           name='softmax', inputs=['dense', 'target_labels'], merge_mode='concat')

m.add_output(name='output', input='softmax')
m.compile('RMSprop', {'output': 'categorical_crossentropy'})

left_context = 3
if train_model:
    batch_X, batch_y = [], []
    for seq in tokenizer.texts_to_sequences_generator(text_generator()):
        start_idx, end_idx = 0, left_context
        while end_idx < len(seq):
            batch_y.append([seq[end_idx]])
            slice_ = np.zeros(max_features)
            for idx in seq[start_idx: end_idx-1]:
                slice_[idx] += 1
            start_idx += 1
            end_idx += 1
            batch_X.append(slice_)
        if len(batch_X) >= 1000:
            batch_X = np.asarray(batch_X)
            batch_y = np.asarray(batch_y)
            matrix_batch_Y = np_utils.to_categorical(batch_y, nb_classes=max_features)
            flat_batch_Y = np.asarray([[1] for i in range(batch_y.shape[0])], dtype='int8')
            #m.fit({'input': batch_X, 'output': batch_Y},
            m.fit({'input': batch_X, 'target_labels': batch_y, 'output': flat_batch_Y},
                     validation_data=None, validation_split=None, nb_epoch=1,
                     batch_size=100, verbose=1, shuffle=True)
            predictions = m.predict({'input': batch_X, 'target_labels': batch_y, 'output': matrix_batch_Y},
                     batch_size =100)
            # reset:
            batch_X, batch_y = [], []





