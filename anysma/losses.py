# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras import backend as K
from keras.losses import *


class Loss(object):
    """Loss base class.
    """
    def __call__(self, y_true, y_pred):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MarginLoss(Loss):
    def __init__(self, lamb=0.5, margin=0.1):
        self.lamb = lamb
        self.margin = margin

    def __call__(self, y_true, y_pred):
        loss = y_true * K.square(K.relu(1 - self.margin - y_pred))\
            + self.lamb * (1 - y_true) * K.square(K.relu(y_pred - self.margin))
        return K.sum(loss, axis=-1)

    def get_config(self):
        return {'lamb': self.lamb,
                'margin': self.margin}


def margin_loss(y_true, y_pred):
    loss_func = MarginLoss()
    return loss_func(y_true, y_pred)


class GlvqLoss(Loss):
    # add flip as prob_trans for real GLVQ cost function
    def __init__(self, squash_func=None):
        self.squash_func = squash_func

    def __call__(self, y_true, y_pred):
        # y_true categorical vector (one-hot value)
        dp = K.squeeze(K.batch_dot(y_true, y_pred, [1, 1]), -1)

        dm = K.max((1 - y_true) * y_pred + y_true * K.min(y_pred, axis=-1, keepdims=True), axis=-1)

        # mu function
        # scale back by unflipping
        loss = (-dp + dm) / \
               (2*K.max(y_pred, axis=-1) - dm - dp + 2*K.min(y_pred, axis=-1))

        if self.squash_func is not None:
            loss = self.squash_func(loss)

        return loss

    def get_config(self):
        return {'squash_func': self.squash_func}


def glvq_loss(y_true, y_pred):
    loss_func = GlvqLoss()
    return loss_func(y_true, y_pred)


def generalized_kullback_leibler_divergence(y_true, y_pred):
    y_true = K.maximum(y_true, K.epsilon())
    y_pred = K.maximum(y_pred, K.epsilon())
    return K.sum(y_true * K.log(y_true / y_pred) - y_true + y_pred, axis=-1)


def itakura_saito_divergence(y_true, y_pred):
    y_true = K.maximum(y_true, K.epsilon())
    y_pred = K.maximum(y_pred, K.epsilon())
    return K.sum(K.log(y_pred / y_true) + y_true / y_pred - 1, axis=-1)


class SpreadLoss(Loss):
    def __init__(self, margin=0.1):
        self.margin = margin

    def __call__(self, y_true, y_pred):
        # mask for the true label
        true_mask = K.cast(K.equal(y_true, 1), dtype=y_pred.dtype)

        # extract correct prediction
        true = K.sum(y_pred * true_mask, axis=-1, keepdims=True)

        # mask the correct class out of the loss vector
        loss = (1 - true_mask) * K.square((K.maximum(0, self.margin - (true - y_pred))))

        return K.sum(loss, axis=-1)

    def get_config(self):
        return {'margin': self.margin}


def spread_loss(y_true, y_pred):
    loss_func = SpreadLoss()
    return loss_func(y_true, y_pred)


# Aliases:


# copy from Keras!
def serialize(loss):
    return serialize_keras_object(loss)


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')


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
        raise ValueError('Could not interpret loss function identifier:', identifier)
