# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras import backend as K

from .linalg_funcs import svd, trace


# ToDo: add tests
def dynamic_routing_squash(vectors, axis=-1, epsilon=K.epsilon()):
    with K.name_scope('squash'):
        squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
        if epsilon == 0:
            return K.sqrt(squared_norm) / (1 + squared_norm) * vectors
        else:
            return K.sqrt(K.maximum(squared_norm, epsilon)) / (1 + squared_norm) * vectors


def orthogonalization(tensors):
    # orthogonalization via polar decomposition
    with K.name_scope('orthogonalization'):
        _, u, v = svd(tensors, full_matrices=False, compute_uv=True)
        if K.ndim(tensors) == 2:
            return K.dot(u, K.permute_dimensions(v, [K.ndim(v) - 1, K.ndim(v) - 2]))
        elif K.ndim(tensors) == 3:
            return K.batch_dot(u,
                               K.permute_dimensions(v, list(range(K.ndim(v) - 2)) + [K.ndim(v) - 1, K.ndim(v) - 2]))
        else:
            raise ValueError("Orthogonalization is not defined for tensors with dimensions other than 2 or 3. You "
                             "provide: " + str(K.ndim(tensors)))


def trace_normalization(tensors, epsilon=K.epsilon()):
    # tensors is assumed as transposed
    # shape(tensors): (optional [proto_number]) x dim_old  x dim_new
    # if symmetrize == False: A = tensors
    # else: A = tensors * tensors.transpose (that's the case for GMLVQ if the matrix is not a precision matrix; omega is
    # assumed as transposed version (!) against the original formula)
    #
    # The  normalization is performed by tensors = tensors / sqrt(trace(A))
    # basically, in a normal learning process the normalization constant should be never zero. But for consistency we
    # introduce this constant too
    with K.name_scope('trace_normalization'):

        if epsilon == 0:
            constant = K.sqrt(trace(tensors, keepdims=True))
        else:
            constant = K.maximum(K.sqrt(trace(tensors, keepdims=True)), epsilon)
        return tensors / constant


def omega_normalization(tensors, epsilon=K.epsilon()):
    with K.name_scope('omega_normalization'):
        ndim = K.ndim(tensors)

        # batch matrices
        if ndim >= 3:
            axes = ndim - 1
            s_tensors = K.batch_dot(tensors, tensors, [axes, axes])
        # non-batch
        else:
            s_tensors = K.dot(tensors, K.transpose(tensors))

        if epsilon == 0:
            constant = K.sqrt(trace(s_tensors, keepdims=True))
        else:
            constant = K.maximum(K.sqrt(trace(s_tensors, keepdims=True)), epsilon)
        return tensors / constant
