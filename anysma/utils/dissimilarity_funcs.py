# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras import backend as K

from .caps_utils import mixed_shape, equal_shape
from .linalg_funcs import p_norm


def minkowski_distance(signals, protos,
                       order_p=2,
                       squared=False,
                       epsilon=K.epsilon()):
    # example Input:
    # shape(signals): batch x proto_number x channels x dim1 x dim2 x ... x dimN
    # shape(protos): proto_number x dim1 x dim2 x ... x dimN
    # it will compute the distance to all prototypes regarding each channel and batch.

    signals = K.permute_dimensions(signals, [0, 2, 1] + list(range(3, K.ndim(signals))))

    if not equal_shape(K.int_shape(signals)[2:], K.int_shape(protos)):
        raise ValueError("The shape of signals[2:] must be equal protos. You provide: signals.shape[2:]="
                         + str(K.int_shape(signals)[2:]) + " != protos.shape=" + str(K.int_shape(protos)))

    with K.name_scope('minkowski_distance'):
        diff = signals - protos
        # sum_axes: [3, 4, ..., N+2]; each dimension after 3 is considered as data dimension
        atom_axes = list(range(3, K.ndim(diff)))

        diss = p_norm(diff, order_p=order_p, axis=atom_axes, squared=squared, keepdims=False, epsilon=epsilon)

        return K.permute_dimensions(diss, [0, 2, 1])


def omega_distance(signals, protos, omegas,
                   order_p=2,
                   squared=False,
                   epsilon=K.epsilon()):
    # Note: omegas is always assumed as transposed!
    # shape(signals): batch x proto_number x channels x dim1 x dim2 x ... x dimN
    # shape(protos): proto_number x dim1 x dim2 x ... x dimN
    # shape(omegas): (optional [proto_number]) x prod(dim1 * dim2 * ... * dimN)  x prod(projected_atom_shape)

    signal_shape = mixed_shape(signals)
    shape = tuple([i if isinstance(i, int) else None for i in signal_shape])

    if not equal_shape((shape[1],) + shape[3:], K.int_shape(protos)):
        raise ValueError("The shape of signals[2:] must be equal protos. You provide: signals.shape[2:]="
                         + str((shape[1],) + shape[3:]) + " != protos.shape=" + str(K.int_shape(protos)))

    with K.name_scope('omega_distance'):
        projected_atom_shape = K.int_shape(omegas)[-1]

        signals = K.permute_dimensions(signals, [0, 2, 1] + list(range(3, len(signal_shape))))
        diff = signals - protos

        if K.ndim(omegas) == 2:
            with K.name_scope('omega_projections'):
                diff = K.reshape(diff, (signal_shape[0] * signal_shape[2], signal_shape[1], -1))
                projected_diff = K.dot(diff, omegas)
                projected_diff = K.reshape(projected_diff,
                                           (signal_shape[0], signal_shape[2], signal_shape[1], projected_atom_shape))

            diss = p_norm(projected_diff, order_p=order_p, axis=3, squared=squared, keepdims=False, epsilon=epsilon)
            return K.permute_dimensions(diss, [0, 2, 1])

        elif K.ndim(omegas) == 3:
            with K.name_scope('omega_projections'):
                diff = K.reshape(diff, (signal_shape[0] * signal_shape[2], signal_shape[1], -1))
                diff = K.permute_dimensions(diff, [1, 0, 2])
                projected_diff = K.batch_dot(diff, omegas)
                projected_diff = K.reshape(projected_diff,
                                           (signal_shape[1], signal_shape[0], signal_shape[2], projected_atom_shape))

            diss = p_norm(projected_diff, order_p=order_p, axis=3, squared=squared, keepdims=False, epsilon=epsilon)
            return K.permute_dimensions(diss, [1, 0, 2])


def tangent_distance(signals, protos, subspaces,
                     squared=False,
                     epsilon=K.epsilon()):
    # Note: subspaces is always assumed as transposed and must be orthogonal!
    # shape(signals): batch x proto_number x channels x dim1 x dim2 x ... x dimN
    # shape(protos): proto_number x dim1 x dim2 x ... x dimN
    # shape(subspaces): (optional [proto_number]) x prod(dim1 * dim2 * ... * dimN)  x prod(projected_atom_shape)

    signal_shape = mixed_shape(signals)
    shape = tuple([i if isinstance(i, int) else None for i in signal_shape])
    subspace_shape = K.int_shape(subspaces)

    if not equal_shape((shape[1],) + shape[3:], K.int_shape(protos)):
        raise ValueError("The shape of signals[2:] must be equal protos. You provide: signals.shape[2:]="
                         + str((shape[1],) + shape[3:]) + " != protos.shape=" + str(K.int_shape(protos)))

    with K.name_scope('tangent_distance'):
        atom_axes = list(range(3, len(signal_shape)))
        signals = K.permute_dimensions(signals, [0, 2, 1] + atom_axes)
        diff = signals - protos

        # global tangent space
        if K.ndim(subspaces) == 2:
            with K.name_scope('projector'):
                projector = K.eye(subspace_shape[-2]) - K.dot(subspaces, K.transpose(subspaces))

            with K.name_scope('tangentspace_projections'):
                diff = K.reshape(diff, (signal_shape[0] * signal_shape[2], signal_shape[1], -1))
                projected_diff = K.dot(diff, projector)
                projected_diff = K.reshape(projected_diff,
                                           (signal_shape[0], signal_shape[2], signal_shape[1]) + signal_shape[3:])

            diss = p_norm(projected_diff, order_p=2, axis=atom_axes, squared=squared, keepdims=False, epsilon=epsilon)
            return K.permute_dimensions(diss, [0, 2, 1])

        # local tangent spaces
        elif K.ndim(subspaces) == 3:
            with K.name_scope('projector'):
                projector = K.eye(subspace_shape[-2]) - K.batch_dot(subspaces, subspaces, [2, 2])

            with K.name_scope('tangentspace_projections'):
                diff = K.reshape(diff, (signal_shape[0] * signal_shape[2], signal_shape[1], -1))
                diff = K.permute_dimensions(diff, [1, 0, 2])
                projected_diff = K.batch_dot(diff, projector)
                projected_diff = K.reshape(projected_diff,
                                           (signal_shape[1], signal_shape[0], signal_shape[2]) + signal_shape[3:])

            diss = p_norm(projected_diff, order_p=2, axis=atom_axes, squared=squared, keepdims=False, epsilon=epsilon)
            return K.permute_dimensions(diss, [1, 0, 2])


def restricted_tangent_distance(signals, protos, subspaces, bounds,
                                squared=False,
                                epsilon=K.epsilon()):
    # Note: subspaces is always assumed as transposed and must be orthogonal!
    # shape(signals): batch x proto_number x channels x dim1 x dim2 x ... x dimN
    # shape(protos): proto_number x dim1 x dim2 x ... x dimN
    # shape(subspaces): (optional [proto_number]) x prod(dim1 * dim2 * ... * dimN)  x prod(projected_atom_shape)

    signal_shape = mixed_shape(signals)
    shape = tuple([i if isinstance(i, int) else None for i in signal_shape])

    if not equal_shape((shape[1],) + shape[3:], K.int_shape(protos)):
        raise ValueError("The shape of signals[2:] must be equal protos. You provide: signals.shape[2:]="
                         + str((shape[1],) + shape[3:]) + " != protos.shape=" + str(K.int_shape(protos)))

    with K.name_scope('tangent_distance'):
        signal_shape = mixed_shape(signals)
        atom_axes = list(range(3, len(signal_shape)))

        signals = K.permute_dimensions(signals, [0, 2, 1] + atom_axes)
        diff = signals - protos

        if K.ndim(subspaces) == 2:
            with K.name_scope('tangentspace_parameters'):
                diff = K.reshape(diff, (signal_shape[0] * signal_shape[2], signal_shape[1], -1))
                params = K.dot(diff, subspaces)

            with K.name_scope('restricted_parameters'):
                in_space = K.cast(K.less_equal(K.abs(params), bounds), dtype=params.dtype)
                params = in_space * params + K.sign(params) * (1 - in_space) * bounds

            with K.name_scope('tangentspace_projections'):
                projected_diff = K.dot(params, K.transpose(subspaces))
                projected_diff = K.reshape(projected_diff,
                                           (signal_shape[0], signal_shape[2], signal_shape[1]) + signal_shape[3:])

                matching_protos = protos + projected_diff

            diss = p_norm(signals - matching_protos,
                          order_p=2, axis=atom_axes, squared=squared, keepdims=False, epsilon=epsilon)
            return K.permute_dimensions(diss, [0, 2, 1])

        else:  # local or capsule_wise
            with K.name_scope('tangentspace_parameters'):
                diff = K.reshape(diff, (signal_shape[0] * signal_shape[2], signal_shape[1], -1))
                diff = K.permute_dimensions(diff, [1, 0, 2])
                params = K.batch_dot(diff, subspaces)

            with K.name_scope('restricted_parameters'):
                bounds = K.expand_dims(bounds, 1)
                in_space = K.cast(K.less_equal(K.abs(params), bounds), dtype=params.dtype)
                params = in_space * params + K.sign(params) * (1 - in_space) * bounds

            with K.name_scope('tangentspace_projections'):
                projected_diff = K.batch_dot(params, subspaces, [2, 2])
                projected_diff = K.permute_dimensions(projected_diff, [1, 0, 2])
                projected_diff = K.reshape(projected_diff,
                                           (signal_shape[0], signal_shape[2], signal_shape[1]) + signal_shape[3:])

                matching_protos = protos + projected_diff

            diss = p_norm(signals - matching_protos,
                          order_p=2, axis=atom_axes, squared=squared, keepdims=False, epsilon=epsilon)
            return K.permute_dimensions(diss, [0, 2, 1])
