# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import six
from keras import backend as K
from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object

import numpy as np


# each function should exist as camel-case (class) and snake-case (function). It's important that
# a class is always camel-cass and a function is snake-case. Otherwise deserialization fails.
class ProbabilityTransformation(object):
    """ProbabilityTransformation base class: all prob_trans inherit from this class.
    """
    # all functions must have the parameter normalization to use it automatically for regression
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, tensor):
        raise NotImplementedError

    def get_config(self):
        return {'axis': self.axis}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def softmax(tensors, axis=-1, max_stabilization=True):
    with K.name_scope('softmax'):
        if max_stabilization:
            tensors = tensors - K.max(tensors, axis=axis, keepdims=True)

        tensors = K.exp(tensors)

        constant = _normalization_constant(tensors, axis, 0)

        return tensors / constant


class Softmax(ProbabilityTransformation):
    def __init__(self,
                 axis=-1,
                 max_stabilization=True):
        self.max_stabilization = max_stabilization

        super(Softmax, self).__init__(axis=axis)

    def __call__(self, tensors):
        return softmax(tensors,
                       axis=self.axis,
                       max_stabilization=self.max_stabilization)

    def get_config(self):
        config = {'max_stabilization': self.max_stabilization}
        super_config = super(Softmax, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


def neg_softmax(tensors, axis=-1, max_stabilization=True):
    with K.name_scope('neg_softmax'):
        tensors = -tensors
        return softmax(tensors,
                       axis=axis,
                       max_stabilization=max_stabilization)


class NegSoftmax(Softmax):
    def __call__(self, tensors):
        return neg_softmax(tensors,
                           axis=self.axis,
                           max_stabilization=self.max_stabilization)


def neg_exp(tensors, axis=-1, max_normalization=False):
    # works only if all numbers are positive and small number means high probability
    # the result is a multi-class label vector
    with K.name_scope('neg_exp'):
        tensors = -tensors
        if max_normalization:
            tensors = tensors - K.max(tensors, axis=axis, keepdims=True)

        return K.exp(tensors)


class NegExp(ProbabilityTransformation):
    def __init__(self,
                 axis=-1,
                 max_normalization=False):
        self.max_normalization = max_normalization

        super(NegExp, self).__init__(axis=axis)

    def __call__(self, tensors):
        return neg_exp(tensors, axis=self.axis, max_normalization=self.max_normalization)

    def get_config(self):
        config = {'max_normalization': self.max_normalization}
        super_config = super(NegExp, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


def flipmax(tensors, axis=-1, lower_bound=K.epsilon()):
    # works only if all numbers are positive and small number means high probability
    # flip the order of the elements
    # vectors = [3, 1, 2, 4] --> [2, 4, 3, 1]
    if lower_bound <= 0:
        raise ValueError("The lower bound must be greater than zero. Otherwise the computation is not stable.")
    with K.name_scope('flipmax'):
        tensors = K.max(tensors, axis=axis, keepdims=True) - tensors + \
                  K.min(tensors, axis=axis, keepdims=True) + lower_bound

        constant = _normalization_constant(tensors, axis, 0)

        return tensors / constant


class Flipmax(ProbabilityTransformation):
    def __init__(self,
                 axis=-1,
                 lower_bound=K.epsilon()):
        self.lower_bound = lower_bound
        if lower_bound <= 0:
            raise ValueError("The lower bound must be greater or equal zero. Otherwise the computation is not "
                             "stable.")

        super(Flipmax, self).__init__(axis=axis)

    def __call__(self, tensors):
        return flipmax(tensors, axis=self.axis, lower_bound=self.lower_bound)

    def get_config(self):
        config = {'lower_bound': self.lower_bound}
        super_config = super(Flipmax, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


def flip(tensors, axis=-1, lower_bound=K.epsilon()):
    # works only if all numbers are positive and small number means high probability
    # flip the order of the elements
    # vectors = [3, 1, 2, 4] --> [2, 4, 3, 1]
    if lower_bound <= 0:
        raise ValueError("The lower bound must be greater than zero.")
    with K.name_scope('flip'):
        max_v = K.max(tensors, axis=axis, keepdims=True) + lower_bound
        tensors = max_v - tensors + K.min(tensors, axis=axis, keepdims=True)

        return tensors / max_v


class Flip(ProbabilityTransformation):
    def __init__(self,
                 axis=-1,
                 lower_bound=K.epsilon()):
        self.lower_bound = lower_bound
        if lower_bound <= 0:
            raise ValueError("The lower bound must be greater than zero.")

        super(Flip, self).__init__(axis=axis)

    def __call__(self, tensors):
        return flip(tensors, axis=self.axis, lower_bound=self.lower_bound)

    def get_config(self):
        config = {'lower_bound': self.lower_bound}
        super_config = super(Flip, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


def margin_flip(tensors, axis=-1,  margin=0.0001, lower_bound=K.epsilon()):
    if lower_bound <= 0:
        raise ValueError("The lower bound must be greater than zero.")
    with K.name_scope('margin_flip'):
        max_v = K.max(tensors, axis=axis, keepdims=True) + lower_bound
        tensors = K.minimum(max_v - tensors + margin, max_v)

        return tensors / max_v


class MarginFlip(ProbabilityTransformation):
    def __init__(self,
                 axis=-1,
                 margin=0.0001,
                 lower_bound=K.epsilon()):
        self.margin = margin
        self.lower_bound = lower_bound
        if lower_bound <= 0:
            raise ValueError("The lower bound must be greater than zero.")

        super(MarginFlip, self).__init__(axis=axis)

    def __call__(self, tensors):
        return margin_flip(tensors, axis=self.axis, margin=self.margin, lower_bound=self.lower_bound)

    def get_config(self):
        config = {'lower_bound': self.lower_bound,
                  'margin': self.margin}
        super_config = super(MarginFlip, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


def margin_log_flip(tensors, axis=-1, margin=0.0001, lower_bound=K.epsilon()):
    # works only if all numbers are positive and small number means high probability
    # flip the order of the elements
    # vectors = [3, 1, 2, 4] --> [2, 4, 3, 1]
    if lower_bound <= 0:
        raise ValueError("The lower bound must be greater than zero.")
    with K.name_scope('log_flip'):
        tensors = K.log(tensors + 1. + lower_bound)
        max_v = K.max(tensors, axis=axis, keepdims=True)
        tensors = K.minimum(max_v - tensors + margin, max_v)

        return tensors / max_v


class MarginLogFlip(ProbabilityTransformation):
    def __init__(self,
                 axis=-1,
                 margin=0.0001,
                 lower_bound=K.epsilon()):
        self.margin = margin
        self.lower_bound = lower_bound
        if lower_bound <= 0:
            raise ValueError("The lower bound must be greater than zero.")

        super(MarginLogFlip, self).__init__(axis=axis)

    def __call__(self, tensors):
        return margin_log_flip(tensors, axis=self.axis, margin=self.margin, lower_bound=self.lower_bound)

    def get_config(self):
        config = {'lower_bound': self.lower_bound,
                  'margin': self.margin}
        super_config = super(MarginLogFlip, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


def _normalization_constant(tensors, axis, epsilon):
    with K.name_scope('normalization_constant'):
        # normalization constant should be always != 0
        if epsilon == 0:
            constant = K.sum(tensors, axis=axis, keepdims=True)
        else:
            constant = K.maximum(K.sum(tensors, axis=axis, keepdims=True), epsilon)

        return constant


def serialize(probability_transformation):
    return serialize_keras_object(probability_transformation)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='probability_transformation')


def get(identifier):
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        identifier = str(identifier)

        if identifier.islower():
            return deserialize(identifier)
        else:
            config = {'class_name': identifier, 'config': {}}
            return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret probability transformation identifier: ' + str(identifier))
