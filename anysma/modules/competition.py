# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import initializers
from keras.engine.topology import InputSpec
from keras import backend as K

from ..utils.caps_utils import mixed_shape
from .. import probability_transformations as prob_trans

from ..capsule import Module
from .. import constraints, regularizers


# Todo: think about adding epsilon as parameter
class NearestCompetition(Module):
    def __init__(self,
                 use_for_loop=True,
                 **kwargs):
        self.use_for_loop = use_for_loop
        # be sure to call this somewhere
        super(NearestCompetition, self).__init__(module_input=True,
                                                 module_output=True,
                                                 need_capsule_params=True,
                                                 **kwargs)

    def _build(self, input_shape):
        if input_shape[0][1] != self.proto_number:
            raise ValueError('The capsule number provided by input_shape is not equal the self.proto_number: '
                             'input_shape[0][1]=' + str(input_shape[0][1]) + ' != ' +
                             'self.proto_number=' + str(self.proto_number) + ". Maybe you forgot to call a routing"
                             " module.")
        if input_shape[1][1] != self.proto_number:
            raise ValueError('The prototype number provided by input_shape is not equal the self.proto_number: '
                             'input_shape[1][1]=' + str(input_shape[1][1]) + ' != ' +
                             'self.proto_number=' + str(self.proto_number))
        if len(input_shape[1]) != 2:
            raise ValueError("The dissimilarity vector must be of length two (batch, dissimilarities per prototype). "
                             "You provide: " + str(len(input_shape[1])) + ". Maybe you forgot to call a routing "
                             "module.")

        self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                           InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        if self.proto_number == self.capsule_number:
            return inputs
        else:
            signals = inputs[0]
            diss = inputs[1]
            signal_shape = mixed_shape(signals)

            if self.use_for_loop:
                diss_stack = []
                signals_stack = []
                sub_idx = None
                with K.name_scope('for_loop'):
                    for p in self._proto_distrib:
                        with K.name_scope('compute_slices'):
                            diss_ = diss[:, p[0]:(p[-1]+1)]
                            signals_ = K.reshape(signals[:, p[0]:(p[-1]+1), :],
                                                 [signal_shape[0] * len(p)] + list(signal_shape[2:]))
                        with K.name_scope('competition'):
                            if len(p) > 1:
                                with K.name_scope('competition_indices'):
                                    argmin_idx = K.argmin(diss_, axis=-1)
                                    if sub_idx is None:
                                        sub_idx = K.arange(0, signal_shape[0], dtype=argmin_idx.dtype)
                                    argmin_idx = argmin_idx + len(p) * sub_idx

                                with K.name_scope('dissimilarity_competition'):
                                    diss_stack.append(K.expand_dims(K.gather(K.flatten(diss_), argmin_idx), -1))

                                with K.name_scope('signal_competition'):
                                    signals_stack.append(K.gather(signals_, argmin_idx))
                            else:
                                diss_stack.append(diss_)
                                signals_stack.append(signals_)

                diss = K.concatenate(diss_stack, 1)

                with K.name_scope('signal_concatenation'):
                    signals = K.concatenate(signals_stack, 1)
                    signals = K.reshape(signals, [signal_shape[0], self.capsule_number] + list(signal_shape[2:]))

            else:
                with K.name_scope('dissimilarity_preprocessing'):
                    # extend if it is not equally distributed
                    if not self._equally_distributed:
                        # permute to first dimension is prototype (protos x batch)
                        diss = K.permute_dimensions(diss, [1, 0])
                        # gather regarding extension (preparing for reshape to block)
                        diss = K.gather(diss, self._proto_extension)
                        # permute back (max_proto_number x (max_proto_number * batch))
                        diss = K.permute_dimensions(diss, [1, 0])

                    # reshape to block form
                    diss = K.reshape(diss, [signal_shape[0] * self.capsule_number, self._max_proto_number_in_capsule])

                with K.name_scope('competition_indices'):
                    # get minimal idx in each class and batch for element selection in diss and signals
                    argmin_idx = K.argmin(diss, axis=-1)
                    argmin_idx = argmin_idx + self._max_proto_number_in_capsule * \
                                 K.arange(0, signal_shape[0] * self.capsule_number, dtype=argmin_idx.dtype)

                with K.name_scope('dissimilarity_competition'):
                    # get minimal values in the form (batch x capsule)
                    diss = K.gather(K.flatten(diss), argmin_idx)
                    diss = K.reshape(diss, [signal_shape[0], self.capsule_number])

                with K.name_scope('signal_preprocessing'):
                    # apply the same steps as above for signals
                    # get signals in: (batch x protos x dim1 x ... x dimN) --> out: (batch x capsule x dim1 x ... x dimN)
                    # extend if is not equally distributed
                    if not self._equally_distributed:
                        signals = K.permute_dimensions(signals, [1, 0] + list(range(2, len(signal_shape))))
                        signals = K.gather(signals, self._proto_extension)
                        signals = K.permute_dimensions(signals, [1, 0] + list(range(2, len(signal_shape))))

                    signals = K.reshape(signals,
                                        [signal_shape[0] * self.capsule_number * self._max_proto_number_in_capsule]
                                        + list(signal_shape[2:]))

                with K.name_scope('signal_competition'):
                    signals = K.gather(signals, argmin_idx)
                    signals = K.reshape(signals, [signal_shape[0], self.capsule_number] + list(signal_shape[2:]))

            return {0: signals, 1: diss}

    def _compute_output_shape(self, input_shape):
        signals = list(input_shape[0])
        diss = list(input_shape[1])

        signals[1] = self.capsule_number
        diss[1] = self.capsule_number
        return [tuple(signals), tuple(diss)]

    def get_config(self):
        config = {'use_for_loop': self.use_for_loop}
        super_config = super(NearestCompetition, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class GibbsCompetition(Module):
    # Todo: Test of Gibbs with scaling
    def __init__(self,
                 beta_initializer='ones',
                 beta_regularizer=None,
                 beta_constraint='NonNeg',
                 use_for_loop=True,
                 **kwargs):

        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.beta = None

        self.use_for_loop = use_for_loop

        # be sure to call this somewhere
        super(GibbsCompetition, self).__init__(module_input=True,
                                               module_output=True,
                                               need_capsule_params=True,
                                               **kwargs)

    def _build(self, input_shape):
        if not self.built:
            if input_shape[0][1] != self.proto_number:
                raise ValueError('The capsule number provided by input_shape is not equal the self.proto_number: '
                                 'input_shape[0][1]=' + str(input_shape[0][1]) + ' != ' +
                                 'self.capsule_number=' + str(self.proto_number) + ". Maybe you forgot to call a "
                                 "routing module.")
            if input_shape[1][1] != self.proto_number:
                raise ValueError('The prototype number provided by input_shape is not equal the self.proto_number: '
                                 'input_shape[1][1]=' + str(input_shape[1][1]) + ' != ' +
                                 'self.proto_number=' + str(self.proto_number))
            if len(input_shape[1]) != 2:
                raise ValueError("The dissimilarity vector must be of length two (batch, dissimilarities per "
                                 "prototype). You provide: " + str(len(input_shape[1])) + ". Maybe you forgot to call "
                                 "a routing module.")

            self.beta = self.add_weight(shape=(self.capsule_number,),
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint,
                                        name='beta')

            self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                               InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        if self.proto_number == self.capsule_number:
            return inputs

        else:
            signals = inputs[0]
            diss = inputs[1]

            signal_shape = None

            # signal.shape: (batch, proto_num, caps_dim1, ..., caps_dimN)
            if self.input_spec[0].ndim > 3:
                signal_shape = mixed_shape(signals)
                signals = K.reshape(signals, signal_shape[0:2] + (-1,))

            if not self._equally_distributed:
                if self.use_for_loop:
                    signals_stack = []
                    diss_stack = []
                    with K.name_scope('for_loop'):
                        for i, p in enumerate(self._proto_distrib):
                            with K.name_scope('compute_slices'):
                                diss_ = diss[:, p[0]:(p[-1]+1)]
                                signals_ = signals[:, p[0]:(p[-1] + 1), :]

                            if len(p) > 1:
                                with K.name_scope('competition_probabilities'):
                                    coefficients = prob_trans.neg_softmax(diss_ * self.beta[i],
                                                                          axis=-1, max_stabilization=True)

                                with K.name_scope('signal_competition'):
                                    signals_stack.append(K.expand_dims(K.batch_dot(coefficients, signals_, [1, 1]), 1))

                                with K.name_scope('dissimilarity_competition'):
                                    diss_stack.append(K.batch_dot(coefficients, diss_, [1, 1]))
                            else:
                                signals_stack.append(signals_)
                                diss_stack.append(diss_)

                    signals = K.concatenate(signals_stack, axis=1)
                    diss = K.concatenate(diss_stack, axis=-1)
                else:
                    extension_idx = []
                    for i in self._proto_extension:
                        if i not in extension_idx:
                            extension_idx.append(i)
                        else:
                            extension_idx.append(max(self._proto_extension)+1)

                    batch_size = K.shape(signals)[0] if signal_shape is None else signal_shape[0]
                    # reshape to block
                    with K.name_scope('competition_probabilities'):
                        with K.name_scope('neg_softmax'):
                            with K.name_scope('coefficients'):
                                beta = K.gather(self.beta, self._capsule_extension)
                                coefficients = -diss * beta
                                # max stabilization
                                coefficients = coefficients - K.max(coefficients, axis=-1, keepdims=True)
                                coefficients = K.exp(coefficients)
                                coefficients = K.concatenate([coefficients,
                                                              K.zeros_like(coefficients[:, 0:1])], axis=-1)
                                coefficients = K.transpose(coefficients)
                                coefficients = K.gather(coefficients, extension_idx)
                                coefficients = K.transpose(coefficients)
                                coefficients = K.reshape(coefficients,
                                                         [batch_size, self.capsule_number,
                                                          self._max_proto_number_in_capsule])
                            # could never be a zero division
                            with K.name_scope('normalization_constant'):
                                constant = K.sum(coefficients, axis=-1, keepdims=True)

                            probs = coefficients / constant

                    with K.name_scope('dissimilarity_preprocessing'):
                        diss = K.transpose(diss)
                        diss = K.gather(diss, self._proto_extension)
                        diss = K.transpose(diss)
                        diss = K.reshape(diss,
                                         [batch_size, self.capsule_number, self._max_proto_number_in_capsule])

                    with K.name_scope('dissimilarity_competition'):
                        diss = K.squeeze(K.batch_dot(probs, K.expand_dims(diss), [2, 2]), -1)

                    with K.name_scope('signal_preprocessing'):
                        signals = K.permute_dimensions(signals, [1, 0, 2])
                        signals = K.gather(signals, self._proto_extension)
                        signals = K.permute_dimensions(signals, [1, 0, 2])
                        signals = K.reshape(signals,
                                            [batch_size, self.capsule_number, self._max_proto_number_in_capsule, -1])

                    with K.name_scope('signal_competition'):
                        signals = K.batch_dot(probs, signals, [2, 2])

            else:
                batch_size = K.shape(signals)[0] if signal_shape is None else signal_shape[0]
                diss = K.reshape(diss, [batch_size, self.capsule_number, self._max_proto_number_in_capsule])

                with K.name_scope('competition_probabilities'):
                    coefficients = prob_trans.neg_softmax(diss * K.expand_dims(self.beta, -1),
                                                          axis=-1, max_stabilization=True)

                with K.name_scope('signal_competition'):
                    signals = K.reshape(signals,
                                        [batch_size, self.capsule_number, self._max_proto_number_in_capsule, -1])
                    signals = K.batch_dot(coefficients, signals, [2, 2])

                with K.name_scope('dissimilarity_competition'):
                    diss = K.squeeze(K.batch_dot(coefficients, K.expand_dims(diss), [2, 2]), -1)

            if self.input_spec[0].ndim > 3:
                signals = K.reshape(signals, [signal_shape[0], self.capsule_number] + list(signal_shape[2:]))

            return {0: signals, 1: diss}

    def _compute_output_shape(self, input_shape):
        signals = list(input_shape[0])
        diss = list(input_shape[1])

        signals[1] = self.capsule_number
        diss[1] = self.capsule_number
        return [tuple(signals), tuple(diss)]

    def get_config(self):
        config = {'beta_initializer': initializers.serialize(self.beta_initializer),
                  'beta_regularizer': regularizers.serialize(self.beta_regularizer),
                  'beta_constraint': constraints.serialize(self.beta_constraint),
                  'use_for_loop': self.use_for_loop}
        super_config = super(GibbsCompetition, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class Competition(Module):
    def __init__(self,
                 probability_transformation='flipmax',
                 **kwargs):

        self.probability_transformation = prob_trans.get(probability_transformation)

        # be sure to call this somewhere
        super(Competition, self).__init__(module_input=True,
                                          module_output=True,
                                          need_capsule_params=True,
                                          **kwargs)

    def _build(self, input_shape):
        if input_shape[0][1] != self.proto_number:
            raise ValueError('The capsule number provided by input_shape is not equal the self.proto_number: '
                             'input_shape[0][1]=' + str(input_shape[0][1]) + ' != ' +
                             'self.capsule_number=' + str(self.proto_number) + ". Maybe you forgot to call a routing"
                             " module.")
        if input_shape[1][1] != self.proto_number:
            raise ValueError('The prototype number provided by input_shape is not equal the self.proto_number: '
                             'input_shape[1][1]=' + str(input_shape[1][1]) + ' != ' +
                             'self.proto_number=' + str(self.proto_number))
        if len(input_shape[1]) != 2:
            raise ValueError("The dissimilarity vector must be of length two (batch, dissimilarities per prototype). "
                             "You provide: " + str(len(input_shape[1])) + ". Maybe you forgot to call a routing "
                             "module.")

        self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                           InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        if self.proto_number == self.capsule_number:
            return inputs

        else:
            signals = inputs[0]
            diss = inputs[1]

            signal_shape = None

            # signal.shape: (batch, proto_num, caps_dim1, ..., caps_dimN)
            if self.input_spec[0].ndim > 3:
                signal_shape = mixed_shape(signals)
                signals = K.reshape(signals, signal_shape[0:2] + (-1,))

            if not self._equally_distributed:
                # we can't define this without a for loop (due to the probability transformation)
                signals_stack = []
                diss_stack = []
                with K.name_scope('for_loop'):
                    for p in self._proto_distrib:
                        with K.name_scope('compute_slices'):
                            diss_ = diss[:, p[0]:(p[-1]+1)]
                            signals_ = signals[:, p[0]:(p[-1] + 1), :]

                        if len(p) > 1:
                            with K.name_scope('competition_probabilities'):
                                coefficients = self.probability_transformation(diss_)

                            with K.name_scope('signal_competition'):
                                signals_stack.append(K.expand_dims(K.batch_dot(coefficients, signals_, [1, 1]), 1))

                            with K.name_scope('dissimilarity_competition'):
                                diss_stack.append(K.batch_dot(coefficients, diss_, [1, 1]))
                        else:
                            signals_stack.append(signals_)
                            diss_stack.append(diss_)

                signals = K.concatenate(signals_stack, axis=1)
                diss = K.concatenate(diss_stack, axis=-1)

            else:
                batch_size = K.shape(signals)[0] if signal_shape is None else signal_shape[0]
                diss = K.reshape(diss, [batch_size, self.capsule_number, self._max_proto_number_in_capsule])

                with K.name_scope('competition_probabilities'):
                    coefficients = self.probability_transformation(diss)

                with K.name_scope('signal_competition'):
                    signals = K.reshape(signals,
                                        [batch_size, self.capsule_number, self._max_proto_number_in_capsule, -1])
                    signals = K.batch_dot(coefficients, signals, [2, 2])

                with K.name_scope('dissimilarity_competition'):
                    diss = K.squeeze(K.batch_dot(coefficients, K.expand_dims(diss), [2, 2]), -1)

            if self.input_spec[0].ndim > 3:
                signals = K.reshape(signals, [signal_shape[0], self.capsule_number] + list(signal_shape[2:]))

            return {0: signals, 1: diss}

    def _compute_output_shape(self, input_shape):
        signals = list(input_shape[0])
        diss = list(input_shape[1])

        signals[1] = self.capsule_number
        diss[1] = self.capsule_number
        return [tuple(signals), tuple(diss)]

    def get_config(self):
        config = {'probability_transformation': prob_trans.serialize(self.probability_transformation)}
        super_config = super(Competition, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))
