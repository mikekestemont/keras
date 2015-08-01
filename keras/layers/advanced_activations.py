from ..layers.core import Layer, MaskedLayer
from ..utils.theano_utils import shared_zeros, shared_ones, sharedX
from .. import activations, initializations, regularizers, constraints
import theano
import theano.tensor as T
import numpy as np


class HierarchicalSoftmax(Layer):
    '''
        Implements a 2-level hierarchical softmax to speed up
        classification problems with a large number of output classes.
        Adapted to run on both GPU and CPU from lisa-groundhog:
        https://github.com/lisa-groundhog/GroundHog/blob/master/groundhog/layers/cost_layers.py
    '''
    def __init__(self, input_dim, output_dim, init='glorot_uniform', activation='relu',
                 weights=None, name=None, W_regularizer=None, b_regularizer=None,
                 activity_regularizer=None, W_constraint=None, b_constraint=None):

        super(HierarchicalSoftmax, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.matrix()

        # level 1:
        self.level1_dim = np.ceil(np.sqrt(self.output_dim)).astype('int64')
        self.W1 = self.init((self.input_dim, self.level1_dim))
        self.b1 = shared_zeros((self.level1_dim))

        # level 2:
        self.level2_dim = np.ceil(self.output_dim/float(self.level1_dim)).astype('int64')
        self.W2 = self.init((self.input_dim, self.level2_dim))
        self.b2 = shared_zeros((self.level2_dim))

        self.params = [self.W1, self.b1, self.W2, self.b2]

        # input params for regularization and constraints apply to both levels:
        self.regularizers = []
        
        self.W1_regularizer = regularizers.get(W_regularizer)
        self.W2_regularizer = regularizers.get(W_regularizer)

        if self.W1_regularizer:
            self.W1_regularizer.set_param(self.W1)
            self.regularizers.append(self.W1_regularizer)
            self.W2_regularizer.set_param(self.W2)
            self.regularizers.append(self.W2_regularizer)

        self.b1_regularizer = regularizers.get(b_regularizer)
        self.b2_regularizer = regularizers.get(b_regularizer)

        if self.b1_regularizer:
            self.b1_regularizer.set_param(self.b1)
            self.regularizers.append(self.b1_regularizer)
            self.b2_regularizer.set_param(self.b2)
            self.regularizers.append(self.b2_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.W1_constraint = constraints.get(W_constraint)
        self.W2_constraint = constraints.get(W_constraint)
        self.b1_constraint = constraints.get(b_constraint)
        self.b2_constraint = constraints.get(b_constraint)

        self.constraints = [self.W1_constraint, self.b1_constraint, self.W2_constraint, self.b2_constraint]
        
        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

    def set_name(self, name):
        self.W1.name = '%s_W1' % name
        self.b1.name = '%s_b1' % name
        self.W2.name = '%s_W2' % name
        self.b2.name = '%s_b2' % name

    def get_output(self, train=False):

        X = self.get_input(train)

        # We assume that, on a previous merge layer (with mode="concat"),
        # the true labels have been appended at the end of the data matrix
        # for training. At test time, these will be ignored.
        
        true_X, target_labels = X[:,:-1], T.cast(X[:,-1], 'int8')

        # propagate input to both levels:
        lev1_activs = self.activation(T.dot(true_X, self.W1) + self.b1)
        lev2_activs = self.activation(T.dot(true_X, self.W2) + self.b2)

        batch_size = true_X.shape[0]
        batch_iter = T.arange(batch_size)

        if train:

            level1_idx = target_labels // self.level1_dim
            level2_idx = target_labels % self.level2_dim

            # select relevant activation column and apply activation:
            lev1_val = lev1_activs[batch_iter, level1_idx]
            lev2_val = lev2_activs[batch_iter, level2_idx]
            
            # multiply both edges cost:
            target_probas = lev1_val * lev2_val

            # output is a matrix of predictions, with dimensionality (batch_size, n_out).
            # Since we only have a probability for the correct label,
            # we assign a probability of zero to all other labels
            output = T.zeros((batch_size, self.output_dim))
            output = T.set_subtensor(output[batch_iter, target_labels], target_probas)

        else:

            def _path_probas(idx):
                lev1_vec, lev2_vec = lev1_activs[idx], lev2_activs[idx]
                result, updates = theano.scan(fn=lambda k, array_: k * array_,
                                              sequences=lev1_vec,
                                              non_sequences=lev2_vec)
                return result.flatten()

            output, updates = theano.scan(fn=_path_probas,
                                       sequences=batch_iter)
            output = T.nnet.softmax(output)
            output = output[:, :self.output_dim] # truncate superfluous paths

        return output
        

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__,
                "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                "b_constraint": self.b_constraint.get_config() if self.b_constraint else None}


class LeakyReLU(MaskedLayer):
    def __init__(self, alpha=0.3):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha

    def get_output(self, train):
        X = self.get_input(train)
        return ((X + abs(X)) / 2.0) + self.alpha * ((X - abs(X)) / 2.0)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "alpha": self.alpha}


class PReLU(MaskedLayer):
    '''
        Reference:
            Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
                http://arxiv.org/pdf/1502.01852v1.pdf
    '''
    def __init__(self, input_shape):
        super(PReLU, self).__init__()
        self.alphas = shared_zeros(input_shape)
        self.params = [self.alphas]
        self.input_shape = input_shape

    def get_output(self, train):
        X = self.get_input(train)
        pos = ((X + abs(X)) / 2.0)
        neg = self.alphas * ((X - abs(X)) / 2.0)
        return pos + neg

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_shape": self.input_shape}


class ParametricSoftplus(MaskedLayer):
    '''
        Parametric Softplus of the form: alpha * (1 + exp(beta * X))

        Reference:
            Inferring Nonlinear Neuronal Computation Based on Physiologically Plausible Inputs
            http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003143
    '''
    def __init__(self, input_shape, alpha_init=0.2, beta_init=5.0):

        super(ParametricSoftplus, self).__init__()
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.alphas = sharedX(alpha_init * np.ones(input_shape))
        self.betas = sharedX(beta_init * np.ones(input_shape))
        self.params = [self.alphas, self.betas]
        self.input_shape = input_shape

    def get_output(self, train):
        X = self.get_input(train)
        return T.nnet.softplus(self.betas * X) * self.alphas

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_shape": self.input_shape,
                "alpha_init": self.alpha_init,
                "beta_init": self.beta_init}
