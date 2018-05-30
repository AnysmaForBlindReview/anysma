# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.constraints import *

from .utils import normalization_funcs as norm_funcs


class TraceNormalization(Constraint):
    def __init__(self, epsilon=K.epsilon()):
        self.epsilon = epsilon

    def __call__(self, w):
        return norm_funcs.trace_normalization(w, epsilon=self.epsilon)

    def get_config(self):
        return {'epsilon': self.epsilon}


class OmegaNormalization(Constraint):
    # can be also applied just for interpretation purposes at the end after training
    # is not compatible with OmegaRegualrizer. Don't apply both on the same time. Numerical instable!
    def __init__(self, epsilon=K.epsilon()):
        self.epsilon = epsilon

    def __call__(self, w):
        return norm_funcs.omega_normalization(w, epsilon=self.epsilon)

    def get_config(self):
        return {'epsilon': self.epsilon}


class Orthogonalization(Constraint):
    def __call__(self, w):
        return norm_funcs.orthogonalization(w)


class Positive(Constraint):
    """Constrains the weights to be positive.
    """

    def __call__(self, w):
        w *= K.cast(K.greater(w, 0.), K.floatx())
        return w

# Aliases


trace_normalization = TraceNormalization
omega_normalization = OmegaNormalization
orthogonalization = Orthogonalization
positive = Positive


def serialize(constraint):
    return serialize_keras_object(constraint)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='constraint')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret constraint identifier: ' + str(identifier))
