# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras import backend as K
from .caps_utils import mixed_shape

import numpy as np


def p_norm(vectors, order_p=2, axis=-1, squared=False, keepdims=False, epsilon=K.epsilon()):
    with K.name_scope('p_norm'):
        # special case 1
        if order_p == 1:
            diss = K.sum(K.abs(vectors), axis=axis, keepdims=keepdims)
            if squared:
                diss = K.square(diss)
        elif order_p == np.inf:
            diss = K.max(K.abs(vectors), axis=axis, keepdims=keepdims)
            if squared:
                diss = K.square(diss)
        # Euclidean distance
        elif order_p == 2:
            diss = K.sum(K.square(vectors), axis=axis, keepdims=keepdims)
            if not squared:
                if epsilon == 0:
                    diss = K.sqrt(diss)
                else:
                    diss = K.sqrt(K.maximum(diss, epsilon))
        elif order_p % 2 == 0:
            diss = K.sum(K.pow(vectors, order_p), axis=axis, keepdims=keepdims)
            if not squared:
                if epsilon == 0:
                    diss = K.pow(diss, 1 / order_p)
                else:
                    diss = K.pow(K.maximum(epsilon, diss), 1 / order_p)
        elif order_p > 1:
            diss = K.sum(K.pow(K.abs(vectors), order_p), axis=axis, keepdims=keepdims)
            if not squared:
                if epsilon == 0:
                    diss = K.pow(diss, 1 / order_p)
                else:
                    diss = K.pow(K.maximum(epsilon, diss), 1 / order_p)
        else:  # < 1
            raise NotImplementedError("The computation of the Minkowski Distance with p<1 could be instable. "
                                      "Therefore, we don't support it so far.")

        return diss


# Todo: add tests for all these functions!
def euclidean_norm(vectors, axis=-1, squared=False, keepdims=False, epsilon=K.epsilon()):
    return p_norm(vectors, order_p=2, axis=axis, squared=squared, keepdims=keepdims, epsilon=epsilon)


def svd(tensors, full_matrices=False, compute_uv=True):
    # we don't handle complex numbers. Thus the conjugated transpose is just the transpose
    # we convert all outputs to the tensorflow standard
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        # return s, u, v
        return tf.svd(tensors, full_matrices=full_matrices, compute_uv=compute_uv)

    elif K.backend() == 'cntk':
        raise NotImplementedError("SVD is not implemented for CNTK")

    elif K.backend() == 'theano':
        import theano as T
        # return u, v, s
        with K.name_scope('svd'):
            if compute_uv:
                u, v, s = T.tensor.nlinalg.svd(tensors, full_matrices=full_matrices, compute_uv=compute_uv)
                return s, u, K.permute_dimensions(v, list(range(K.ndim(v) - 2)) + [K.ndim(v)-1, K.ndim(v) - 2])
            else:
                return T.tensor.nlinalg.svd(tensors, full_matrices=full_matrices, compute_uv=compute_uv)
    else:
        raise NotImplementedError("Unknown backend `" + K.backend() + "`.")


def det(tensors, keepdims=False):
    # we don't handle complex numbers. Thus the conjugated transpose is just the transpose
    # we convert all outputs to the tensorflow standard
    if K.backend() == 'tensorflow':
        import tensorflow as tf

        d = tf.linalg.det(tensors)
        if keepdims:
            d = K.expand_dims(K.expand_dims(d, -1), -1)

        return d

    elif K.backend() == 'cntk':
        raise NotImplementedError("determinant is not implemented for CNTK")

    elif K.backend() == 'theano':
        raise NotImplementedError("determinant is not implemented for CNTK")

    else:
        raise NotImplementedError("Unknown backend `" + K.backend() + "`.")


def trace(tensors, keepdims=True):
    # trace of a squared matrix
    with K.name_scope('trace'):
        shape = mixed_shape(tensors)
        int_shape = K.int_shape(tensors)
        if int_shape[-1] != int_shape[-2] or int_shape[-1] is None or int_shape[-2] is None:
            raise ValueError("The matrix dimension (the two last dimensions) of the tensor must be equivalent to a "
                             "squared matrix. You provide: " + str(int_shape[-2:]) + ". Be sure that you aren't using "
                             "shape inference in case you observe `None`.")
        tensors = K.reshape(tensors, shape[:-2] + (-1,))
        tensors = K.transpose(tensors)
        tensors = K.gather(tensors, list(range(0, int_shape[-1]**2, int_shape[-1]+1)))
        tensors = K.sum(tensors, axis=0, keepdims=keepdims)
        if keepdims:
            # add last dimension to keepdims
            tensors = K.expand_dims(tensors, 0)
        return K.transpose(tensors)


# Aliases:
norm = euclidean_norm
