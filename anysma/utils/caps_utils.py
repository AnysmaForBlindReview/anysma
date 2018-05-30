# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras import backend as K


def dict_to_list(x):
    if isinstance(x, dict):
        return [x[i] for i in x]
    else:
        return x


def list_to_dict(x):
    if isinstance(x, (tuple, list)):
        return {i: x[i] for i in range(len(x))}
    else:
        return x


def mixed_shape(inputs):
    with K.name_scope('mixed_shape'):
        int_shape = list(K.int_shape(inputs))
        tensor_shape = K.shape(inputs)

        for i, s in enumerate(int_shape):
            if s is None:
                int_shape[i] = tensor_shape[i]
        return tuple(int_shape)


def equal_shape(shape_1, shape_2):
    for axis, value in enumerate(shape_1):
        if value is not None and shape_2[axis] not in {value, None}:
            return False
    return True
