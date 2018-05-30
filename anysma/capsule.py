# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.layers import deserialize as deserialize_layer

from .utils.caps_utils import dict_to_list, list_to_dict
from . import constraints, regularizers

import numpy as np

# Todo: think about Masking Layers like capsnet;
# Todo: load initializers, regualrizers, etc. from Keras in anysma and call them if not found in anysma
# Todo: Think about to define a own function for sqrt to define an own gradient function
# Todo: check carefully the use of K.maximum(..., epsilon) stabilization. It could lead to some trouble (remember capsnet)
# Todo: if it is avoidable remove zero dimensions for variables (like beta)
# Todo: remove initializations of weights with one dimensions
# Todo: define CropLayer in caps_utils (input: filter stack, etc.; output: [first_crop, second_crop])
# Todo: add capsule.add() with Flag for atoms/ signals, or diss
# Todo: add flag for tiling_false; We need to support a more efficient computation if there is no need to tile the
# signal regarding the protos (for example if the transformation is global and no proto dependent transformation on
# the signals)


class Module(Layer):
    """Module Abstract Class

    Conventions:
        - internally is each input considered as dict input; the internal preprocessing provides an correct
          connection to sub-function. We need another data type to identify uniquely if a input is from a module or not
        - output and input of real module is always a dict (!) and just that ! The dict exist only outside the module
          for the internal layer registration it is again a list of tensors
        - additional inputs like losses are non-dict an basic keras type. It is hard to trace where keras uses these
          inputs after the storage (also true for masks (!) and shapes, updates, losses) (!)
          ! consider this carefully if you call methods which have parameters from both groups like (add_losses)
        - the shape of inputs are considered as lists (without the dict around). dicts haven't a shape property.
        - the first dimension after batch_size is the channel dimension of the capsule:
          (batch_size, channels, dim1, dim2 , dim3, ..., dimN)

    Abstract body for own module:

class MyModule(Module):
    def __init__(self, output_shape, **kwargs):
        self.output_shape = output_shape
        # be sure to call this somewhere
        super(MyModule, self).__init__(**kwargs)

    def _call(self, inputs, **kwargs):
        # be sure to call this at the first position
        return inputs

    def _build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

     def _compute_mask(self, inputs, masks=None):
        # if needed implement masking here
        return masks

    def _compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        config = {'need_capsule_params': self.need_capsule_params,
                  'layer_init': self._layer_init}
        super_config = super(Module, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))

    """
    def __init__(self,
                 need_capsule_params=True,
                 module_input=None,
                 module_output=None,
                 **kwargs):
        """init method of a module"""
        # register capsule params here! different to Layer params (!)
        self._needed_kwargs = {'_proto_distrib',
                               '_proto_extension',
                               '_capsule_extension',
                               '_max_proto_number_in_capsule',
                               '_equally_distributed',
                               'proto_number',
                               'capsule_number',
                               'owner'}
        self._proto_distrib = None
        self._proto_extension = None
        self._capsule_extension = None
        self._max_proto_number_in_capsule = None
        self._equally_distributed = None
        self.proto_number = None
        self.capsule_number = None
        self.owner = None

        # defines if the signal is dict (module input) or standard keras; is fixed after first module call (except for
        # capsules). It is just fixed automatically if it is None! Since, could be predefined.
        self._module_input = module_input
        self._module_output = module_output

        # if the signal is True all needed_kwargs must be unequal to None: real module!
        # a Module can't change the state during runtime which is given by initialization. Since during the
        # programming we assume that we have access to the params
        # A capsule can change this state during build.
        self._need_capsule_params = need_capsule_params

        # take kwargs, pop Capsule kwargs out and put the base Layer
        kwargs = self._set_params(**kwargs)

        super(Module, self).__init__(**kwargs)

        # default owner is the instance itself
        if self.owner is None:
            self.owner = self.name
        if self.need_capsule_params is None:
            raise ValueError('You must specify if the module is real (True or False). You provide: ' +
                             str(self._need_capsule_params))

    def __call__(self, inputs, **kwargs):
        """Call command: basic layer call with some pre-processing"""
        if not self.built:
            # after a module is called the input type is fixed; it causes problems to make it dynamic! During the
            # model compile of keras the compute_output_shape is could be called. We can lost the reference to the true
            # value if we keep it dynamic. Further if we keep it dynamic we have to handle in all methods both cases:
            # lists, dicts. (see Keras -> topology -> container -> run_internal_graph)
            if self._module_input is None:
                self._module_input = self._is_module_input(inputs)
            # set output type to input type if not pre-defined
            if self._module_output is None:
                self._module_output = self._is_module_input(inputs)
            # after this capsule params can't be changed
            kwargs = self._set_params(**kwargs)
            # self.name = self.name + '_owner_' + self.owner
        else:
            # capsule params are not transmitted; pop out
            kwargs = self._del_needed_kwargs(**kwargs)

        self._is_callable(inputs)

        # compute outputs (outputs is list, tuple or a keras tensor)
        outputs = super(Module, self).__call__(self._to_layer_input(inputs), **kwargs)

        # Todo: check regularizer and provide config support
        # Check activity_regularizer and throw error if the attribute exist:
        if hasattr(self, 'activity_regularizer'):
            raise AttributeError("'activity_regularizer' should not be used for capsules, because there is no support "
                                 "to apply different regularizers over the single output entities. Use "
                                 "'output_regularizers' instead.")

        # Apply output_regularizers if any:
        if hasattr(self, 'output_regularizers') and self.output_regularizers is not None:
            if callable(self.output_regularizers):
                if isinstance(outputs, (list, tuple)):
                    regularization_losses = [self.output_regularizers(x) for x in outputs]
                    self.add_loss(regularization_losses, inputs)
                else:
                    self.add_loss([self.output_regularizers(outputs)], inputs)
            else:
                if isinstance(outputs, (list, tuple)):
                    if len(self.output_regularizers) != len(outputs):
                        raise ValueError("The list of 'output_regularizers' must have the same length as outputs. You "
                                         "provide: len(outputs)=" + str(len(outputs)) + " != len(output_regularizers)="
                                         + str(len(self.output_regularizers)))
                    regularization_losses = [self.output_regularizers[i](x) for i, x in enumerate(outputs)]
                    self.add_loss(regularization_losses, inputs)
                else:
                    if len(self.output_regularizers) != 1:
                        raise ValueError("The list of 'output_regularizers' must have the same length as outputs. You "
                                         "provide: len(outputs)=" + str(1) + " != len(output_regularizers)="
                                         + str(len(self.output_regularizers)))
                    self.add_loss([self.output_regularizers[0](outputs)], inputs)

        return self._to_module_output(outputs, module_output=self._module_output)

    def build(self, input_shape):
        # same structure as Layer
        self._is_callable()
        self._build(input_shape)
        super(Module, self).build(input_shape)

    def _build(self, input_shape):
        # implement your build method here
        pass

    def call(self, inputs, **kwargs):
        was_module_input = self._is_module_input(inputs)
        inputs = self._to_module_input(inputs)
        self._is_callable(inputs)
        outputs = self._call(inputs, **kwargs)
        return self._to_module_output(outputs, module_output=was_module_input)

    def _call(self, inputs, **kwargs):
        # implement your call method here
        return inputs

    def compute_output_shape(self, input_shape):
        self._is_callable()
        return self._compute_output_shape(input_shape)

    def _compute_output_shape(self, input_shape):
        """implement your output_shape inference here"""
        return input_shape

    def compute_mask(self, inputs, mask=None):
        inputs = self._to_module_input(inputs)
        self._is_callable(inputs)
        # It could be possible that we have to transform the signal back (same structure like call())
        return self._compute_mask(inputs, mask)

    def _compute_mask(self, inputs, mask):
        """Computes an output mask tensor.

        Copy of base implementation from Keras.

        # Arguments
            inputs: Tensor or list of tensors.
            mask: Tensor or list of tensors.

        # Returns
            None or a tensor (or list of tensors,
                one per output tensor of the layer).
        """
        if not self.supports_masking:
            if mask is not None:
                if isinstance(mask, list):
                    if any(m is not None for m in mask):
                        raise TypeError('Module ' + self.name + ' does not support masking, but was passed an '
                                        'input_mask: ' + str(mask))
                else:
                    raise TypeError('Module ' + self.name + ' does not support masking, but was passed an input_'
                                    'mask: ' + str(mask))
            # masking not explicitly supported: return None as mask
            return None
        # if masking is explicitly supported, by default
        # carry over the input mask
        return mask

    def _set_params(self, **kwargs):
        """Set needed_kwargs of Capsule by **kwargs and return the reduced kwargs list."""
        for key in self._needed_kwargs:
            if key in kwargs:
                # pop key-value-pairs!
                setattr(self, key, kwargs.pop(key))
        # return modified kwargs
        return kwargs

    def _get_params(self):
        """Get needed_kwargs of Capsule"""
        params = {}
        for key in self._needed_kwargs:
            params.update({key: getattr(self, key)})
        return params

    def _is_callable(self, inputs=None):
        """Check if the module is callable: check if it was pre-processed."""
        if self.need_capsule_params is True:
            for key in self._needed_kwargs:
                if getattr(self, key) is None:
                    raise ValueError('The module ' + self.name + ' is not proper initialized for calls. Some '
                                     'of the parameters are not unequal None (that\'s the convention).  Be sure that '
                                     'the module is assigned to a proper capsule. Parameter ' + key + ' is None. The '
                                     'owner capsule is :' + str(self.owner))
        if self._module_input is None:
            raise ValueError('The input type of the module is not specified. Can\'t process automatically if the input'
                             ' is a module input or not. This is usually the case if you call a module method which'
                             ' is not callable without internal pre-processing.')
        if self._module_output is None:
            raise ValueError('The output type of the module is not specified. Can\'t process automatically if the '
                             'output is a module output or not. This is usually the case if you call a module method '
                             'which is not callable without internal pre-processing.')
        if inputs is not None:
            if self._module_input != self._is_module_input(inputs):
                raise TypeError('The input type of ' + self.name + ' is ' + ('dict (module signal)' if
                                self._module_input else '[list, tuple, tensor]') + ' and you provide '
                                + str(type(inputs)))

        return True

    def _del_needed_kwargs(self, **kwargs):
        """del needed_kwargs to avoid spoiling of the Layer methods"""
        for key in self._needed_kwargs:
            if key in kwargs:
                del kwargs[key]
        return kwargs

    def _to_module_output(self, inputs, module_output):
        """convert a module input (dict) to a layer signal (list or tensor)"""
        if inputs is not None:
            if module_output:
                if not self._is_module_input(inputs):
                    return list_to_dict(inputs)
                else:
                    return inputs
            else:
                if self._is_module_input(inputs):
                    return dict_to_list(inputs)
                else:
                    return inputs
        else:
            return inputs

    def _to_layer_input(self, inputs):
        """transforms the input back to a dict if it was a module signal"""
        if self._module_input:
            if inputs is not None:
                if self._is_module_input(inputs):
                    return dict_to_list(inputs)
                else:
                    return inputs
            else:
                return inputs
        else:
            return inputs

    def _to_module_input(self, inputs):
        """transforms the input back to a dict if it was a module signal"""
        if self._module_input:
            if inputs is not None:
                if not self._is_module_input(inputs):
                    return list_to_dict(inputs)
                else:
                    return inputs
            else:
                return inputs
        else:
            return inputs

    @property
    def need_capsule_params(self):
        """get the flag if it a real module"""
        return self._need_capsule_params

    @staticmethod
    def _is_module_input(inputs):
        return isinstance(inputs, dict)

    def assert_input_compatibility(self, inputs):
        self._is_callable()
        super(Module, self).assert_input_compatibility(self._to_layer_input(inputs))

    def get_input_at(self, node_index):
        self._is_callable()
        inputs = super(Module, self).get_input_at(node_index)
        return self._to_module_output(inputs, self._module_input)

    def get_output_at(self, node_index):
        self._is_callable()
        outputs = super(Module, self).get_output_at(node_index)
        return self._to_module_output(outputs, self._module_output)

    @property
    def input(self):
        self._is_callable()
        inputs = super(Module, self).input
        return self._to_module_output(inputs, self._module_input)

    @property
    def output(self):
        self._is_callable()
        outputs = super(Module, self).output
        return self._to_module_output(outputs, self._module_output)

    def add_loss(self, losses, inputs=None):
        self._is_callable()
        inputs = self._to_layer_input(inputs)
        super(Module, self).add_loss(losses, inputs)

    def add_update(self, updates, inputs=None):
        self._is_callable()
        inputs = self._to_layer_input(inputs)
        super(Module, self).add_update(updates, inputs)

    def get_updates_for(self, inputs):
        self._is_callable()
        inputs = self._to_layer_input(inputs)
        return super(Module, self).get_updates_for(inputs)

    def get_losses_for(self, inputs):
        self._is_callable()
        inputs = self._to_layer_input(inputs)
        return super(Module, self).get_losses_for(inputs)

    def get_config(self):
        config = {'need_capsule_params': self.need_capsule_params,
                  'module_input': self._module_input,
                  'module_output': self._module_output}
        super_config = super(Module, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class _ModuleWrapper(Module):
    """Wrapper to interpret Keras layers as module

    Just a simple wrapper. Should be not used outside of this class. We don't provide direct links to the layer
    properties to avoid sync problems.
    """
    # it is assumed that a layer never deals with dicts of inputs
    def __init__(self,
                 layer,
                 **kwargs):
        self.layer = layer

        # pop is real module if provided
        if 'need_capsule_params' in kwargs:
            del kwargs['need_capsule_params']

        super(_ModuleWrapper, self).__init__(need_capsule_params=False, **kwargs)

    def __call__(self, inputs, **kwargs):
        if not self.built:
            if self._module_input is None:
                self._module_input = self._is_module_input(inputs)
            if self._module_output is None:
                self._module_output = self._is_module_input(inputs)
            # after this capsule params can't be changed
            kwargs = self._set_params(**kwargs)
            # self.layer.name = self.layer.name + '_owner_' + self.owner
        else:
            kwargs = self._del_needed_kwargs(**kwargs)

        self._is_callable(inputs)

        if self._module_input:
            outputs = inputs
            outputs[0] = self.layer.__call__(inputs[0], **kwargs)
        else:
            outputs = self.layer.__call__(inputs, **kwargs)
        return outputs

    def _build(self, input_shape):
        if self._module_input:
            self.layer.build(input_shape[0])
        else:
            self.layer.build(input_shape)

    def _call(self, inputs, **kwargs):
        kwargs = self._del_needed_kwargs(**kwargs)
        if self._module_input:
            return inputs.update({0: self.layer.call(inputs[0], **kwargs)})
        else:
            return self.layer.call(inputs, **kwargs)

    def _compute_output_shape(self, input_shape):
        if self._module_input:
            input_shape[0] = self.layer.compute_output_shape(input_shape[0])
            return input_shape
        else:
            return self.layer.compute_output_shape(input_shape)

    def _compute_mask(self, inputs, mask=None):
        if self._module_input:
            mask[0] = self.layer.compute_mask(inputs[0], mask)
            return mask
        else:
            return self.layer.compute_mask(inputs[0], mask)

    def get_config(self):
        return self.layer.get_config()


class Capsule(Module):
    """Capsule

    Be careful if you iterate through the stack. The list could be not unique (distinguish between list and set)!
    """
    # and which modules methods must be overloaded to work properly
    # provide plot function, like summary
    def __init__(self,
                 prototype_distribution=None,
                 **kwargs):
        # trainable parameters are read via trainable_weights; paras which not occur in this list are non-trainable
        # thus we can control trainable there and consider the whole trainable state of a capsule (!)
        # self.trainable = True

        # store user input for return
        # the Capsule is build by call (all special caps paras are then init)
        self.proto_distribution = prototype_distribution
        self._module_stack = []

        # pop is real module if provided
        if 'need_capsule_params' in kwargs:
            del kwargs['need_capsule_params']

        # call at the beginning avoid conflicts in the setting of params
        # if it is a real module is specified in build with respect to the setting,; thus definition is made
        # during the runtime!
        super(Capsule, self).__init__(need_capsule_params=False, **kwargs)

        # It's up to the modules to support masking, trainable
        self.supports_masking = True
        self.trainable = True

    def __call__(self, inputs, **kwargs):
        """Modification of kears Layer.__call__

        We let all the checks of the input on the module side with the basic layer __call__
        """
        # input type of a capsule can be changed over calls; there is no need to fix it...let the validity check up to
        # the module
        self._module_input = self._is_module_input(inputs)
        # we don't need this parameter but we have to set it: It's up to the modules if the output is dict().
        self._module_output = self._is_module_input(inputs)
        with K.name_scope(self.name):
            if not self.built:
                if isinstance(inputs, (list, tuple, dict)):
                    input_shape = []
                    for key in range(len(inputs)):
                        input_shape.append(K.int_shape(inputs[key]))
                else:
                    input_shape = K.int_shape(inputs)
                self.build(input_shape)
            self._is_callable()
            # set capsule paras in kwargs (overwrite other); this is the reason why a capsule can call a capsule!
            kwargs.update(self._get_params())
            return self.call(inputs, **kwargs)

    def _build(self, input_shape):
        """ None: not needed: module init independent to proto distrib
                tuple (x, y): x num protos per capsule, y capsule number
                list: len(list) = num of capsules, list[i] = protos in i
                int: int is num of capsules one proto per class
                return: nd.array
        """
        if not self.built:
            d = self.proto_distribution
            if d is None:
                distrib = None
                proto_extension = None
                capsule_extension = None

            elif isinstance(d, tuple) and len(d) == 2:
                if d[0] < 1 or d[1] < 1:
                    raise ValueError('The number of prototypes per capsule and the number of capsules must be greater '
                                     'than 0. You provide: ' + str(d))
                distrib = []
                capsule_extension = []
                for i in range(d[1]):
                    distrib.append(list(range(i * d[0], (i+1) * d[0])))
                    capsule_extension.extend(list(i * np.ones(d[0], dtype=int)))
                proto_extension = list(range(int(d[0] * d[1])))

            elif isinstance(d, list) and len(d) > 0:
                # proto_extension = np.array([], dtype=int)
                proto_extension = []
                capsule_extension = []
                distrib = []
                for i, d_ in enumerate(d):
                    if d_ < 1:
                        raise ValueError('The number of prototypes per capsule must be greater '
                                         'than 0. You provide: ' + str(d_) + ' at index: ' + str(i))
                    distrib.append(list(range(sum(d[0:i]), sum(d[0:(i+1)]))))
                    # proto_extension = np.concatenate((proto_extension,
                    #                                   np.arange(sum(d[0:i]), sum(d[0:i + 1]), dtype=int)))
                    # proto_extension = np.concatenate((proto_extension,
                    #                                   proto_extension[-1] * np.ones(max(d) - d[i], dtype=int)))
                    proto_extension.extend(list(range(sum(d[0:i]), sum(d[0:i + 1]))))
                    proto_extension.extend(list(proto_extension[-1] * np.ones(max(d) - d_, dtype=int)))
                    capsule_extension.extend(list(i * np.ones(d_, dtype=int)))

            elif isinstance(d, int):
                if d < 1:
                    raise ValueError('The number of capsules must be greater than 0. You provide:' + str(d))
                distrib = []
                for i in range(d):
                    distrib.append([i])
                proto_extension = list(range(d))
                capsule_extension = list(range(d))

            else:
                raise TypeError("The argument must be a 2D-tuple, list, None or int. You pass : '" + str(d) + "'.")

            self._proto_distrib = distrib
            self._proto_extension = proto_extension
            self._capsule_extension = capsule_extension
            self._max_proto_number_in_capsule = 0 if distrib is None else max([len(x) for x in distrib])
            self.capsule_number = 0 if distrib is None else len(distrib)
            self.proto_number = 0 if distrib is None else sum([len(x) for x in distrib])
            self._equally_distributed = False if proto_extension is None else \
                proto_extension == list(range(self.proto_number))

            # check if the module is real and set the flag
            self._need_capsule_params = True
            params = self._get_params()
            for key in params:
                if params[key] is None:
                    self._need_capsule_params = False
                    break

    def _call(self, inputs, **kwargs):
        """inputs are [vectors, d] or flattened in first capsule"""
        outputs = inputs
        for module in self._module_stack:
            # check if you can call the module
            if self.need_capsule_params is False and module.need_capsule_params is True:
                raise TypeError('The non-real Capsule ' + self.name + ' can\'t call the real module '
                                + module.name + '(type: ' + str(type(module)) + ').')
            outputs = module(outputs, **kwargs)

        return outputs

    def _compute_output_shape(self, input_shape):
        """input shape is just a tuple keras_shape"""
        # let further checks up to modules
        output_shape = input_shape
        # must call  output inference of stack
        for module in self._module_stack:
            output_shape = module.compute_output_shape(output_shape)
        return output_shape

    def _compute_mask(self, inputs, mask=None):
        # which could be an indicator for missing squeeze.
        outputs = inputs
        for module in self._module_stack:
            # check if you can call the module
            if self.need_capsule_params is False and module.need_capsule_params is True:
                raise TypeError('The non-real Capsule ' + self.name + ' can\'t call the real module '
                                + module.name + '(type: ' + str(type(module)) + ').')
            mask = module.compute_mask(outputs, mask)
            outputs = module(outputs, **self._get_params())
        return outputs

    # overload tis method so that is up to the module to pre-process the signal
    def _to_module_output(self, inputs, module_output=False):
        return inputs

    # overload tis method so that it is up to the module to pre-process the signal
    def _to_layer_input(self, inputs):
        return inputs

    # overload tis method so that it is up to the module to pre-process the signal
    def _to_module_input(self, inputs):
        return inputs

    @staticmethod
    def _getattr(module, attr=None):
        # return nested layer or module if None
        if attr is None:
            if isinstance(module, _ModuleWrapper):
                return module.layer
            else:
                return module
        else:
            if isinstance(module, _ModuleWrapper):
                return getattr(module.layer, attr)
            else:
                return getattr(module, attr)

    @staticmethod
    def _setattr(module, attr, value):
        if isinstance(module, _ModuleWrapper):
            setattr(module.layer, attr, value)
        else:
            setattr(module, attr, value)

    @staticmethod
    def _hasattr(module, attr):
        if isinstance(module, _ModuleWrapper):
            return hasattr(module.layer, attr)
        else:
            return hasattr(module, attr)

    def _get_modules(self):
        stack = []
        for module in self._module_stack:
            if stack.count(module) == 0:
                stack.append(module)
        return stack

    def add(self, modules):
        if not isinstance(modules, (list, tuple)):
            modules = [modules]

        for module in modules:
            if not isinstance(module, (Layer, Module)):
                raise TypeError('The added module must be an instance of class Layer or Module. Found: ' + str(module)
                                + ". Maybe you forgot to initialize the module.")
            # check if is a module by checking teh existence of some attributes
            if hasattr(module, '_module_input') and hasattr(module, '_need_capsule_params'):
                self._module_stack.append(module)
            # each type which is not a module is considered as layer type
            else:
                idx = None
                for m in self._get_modules():
                    # check if layer is always added as module
                    if module == self._getattr(m):
                        idx = self._module_stack.index(m)
                        break
                if idx is None:
                    self._module_stack.append(_ModuleWrapper(module))
                else:
                    self._module_stack.append(self._module_stack[idx])
        return self

    def pop(self):
        """Removes the last module in the capsule.

        # Raises
            TypeError: if there are no modules in the capsule.
        """
        # Todo: Program all list functions for Capsule
        if not self._module_stack:
            raise TypeError('There are no modules in the capsule.')
        return self._getattr(self._module_stack.pop())

    @property
    def module_stack(self):
        stack = []
        for module in self._module_stack:
            stack.append(self._getattr(module))
        return stack

    @property
    def modules(self):
        """Get all used modules (not the stack)
        """
        stack = []
        for module in self._get_modules():
            stack.append(self._getattr(module))
        return stack

    def get_module(self, name=None, index=None):
        """Retrieves a module based on either its name (unique) or index.

        Slightly modified copy from Keras.

        Indices are based on the order of adding.

        # Arguments
            name: String, name of module.
            index: Integer, index of module.

        # Returns
            A module instance.

        # Raises
            ValueError: In case of invalid module name or index.
        """
        # It would be unreliable to build a dictionary
        # based on layer names, because names can potentially
        # be changed at any point by the user
        # without the container being notified of it.
        if index is not None:
            if len(self._module_stack) <= index:
                raise ValueError('Was asked to retrieve module at index ' + str(index) + ' but capsule only has ' +
                                 str(len(self._module_stack)) + ' modules.')
            else:
                return self._getattr(self._module_stack[index])
        else:
            if not name:
                raise ValueError('Provide either a module name or module index.')

        for module in self._module_stack:
            if self._getattr(module, 'name') == name:
                return self._getattr(module)

        raise ValueError('No such module: ' + name)

    @property
    def proto_distrib(self):
        return self._proto_distrib

    @property
    def trainable_weights(self):
        """Trainable weights over _get_modules() and not the stack."""
        weights = []
        for module in self._get_modules():
            weights += self._getattr(module, 'trainable_weights')
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        for module in self._get_modules():
            weights += self._getattr(module, 'non_trainable_weights')
        return weights

    def get_weights(self):
        """Retrieves the weights of the capsule.

        # Returns
            A flat list of Numpy arrays.
        """
        weights = []
        for module in self._get_modules():
            weights += self._getattr(module, 'weights')
        return K.batch_get_value(weights)

    def set_weights(self, weights):
        """Sets the weights of the capsule.

        # Arguments
            weights: A list of Numpy arrays with shapes and types matching
                the output of `capsule.get_weights()`.
        """
        tuples = []
        for module in self._get_modules():
            num_param = len(self._getattr(module, 'weights'))
            layer_weights = weights[:num_param]
            for sw, w in zip(self._getattr(module, 'weights'), layer_weights):
                tuples.append((sw, w))
            weights = weights[num_param:]
        K.batch_set_value(tuples)

    def get_input_shape_at(self, node_index):
        return self._getattr(self._module_stack[0], 'get_input_shape_at')(node_index)

    def get_output_shape_at(self, node_index):
        return self._getattr(self._module_stack[-1], 'get_output_shape_at')(node_index)

    def get_input_at(self, node_index):
        return self._getattr(self._module_stack[0], 'get_input_at')(node_index)

    def get_output_at(self, node_index):
        return self._getattr(self._module_stack[-1], 'get_output_at')(node_index)

    def get_input_mask_at(self, node_index):
        return self._getattr(self._module_stack[0], 'get_input_mask_at')(node_index)

    def get_output_mask_at(self, node_index):
        return self._getattr(self._module_stack[-1], 'get_output_mask_at')(node_index)

    @property
    def input(self):
        return self._getattr(self._module_stack[0], 'input')

    @property
    def output(self):
        return self._getattr(self._module_stack[-1], 'output')

    @property
    def input_mask(self):
        return self._getattr(self._module_stack[0], 'input_mask')

    @property
    def output_mask(self):
        return self._getattr(self._module_stack[-1], 'output_mask')

    @property
    def input_shape(self):
        return self._getattr(self._module_stack[0], 'input_shape')

    @property
    def output_shape(self):
        return self._getattr(self._module_stack[-1], 'output_shape')

    def get_config(self):
        config = {'prototype_distribution': self.proto_distribution}
        super_config = super(Capsule, self).get_config()
        config = dict(list(super_config.items()) + list(config.items()))

        stack_config = []
        for module in self._module_stack:
            stack_config.append(self._getattr(module, 'name'))
        config.update({'module_stack': stack_config})

        modules_config = {}
        for module in self._get_modules():
            module_config = {'class_name': (self._getattr(module, '__class__')).__name__,
                             'config': (self._getattr(module, 'get_config'))()}
            modules_config.update({self._getattr(module, 'name'): module_config})
        config.update({'modules': modules_config})

    @classmethod
    def from_config(cls, config):
        modules_config = config.pop('modules')
        modules = {}

        def process_modules():
            # import here; at the beginning leads to errors
            from .modules import globals_modules as globs

            for module_name in modules_config:
                module = deserialize_layer(modules_config[module_name], custom_objects=globs())
                modules.update({module_name: module})

        process_modules()

        stack_config = config.pop('module_stack')
        module_stack = []
        for name in stack_config:
            module_stack.append(modules[name])

        capsule = cls(**config)
        capsule.add(module_stack)
        return capsule

    def summary(self, line_length=None, positions=None, print_fn=print):
        """Prints a string summary of the network.

        Copy from Keras with slight modifications.

        # Arguments
            line_length: Total length of printed lines
                (e.g. set this to adapt the display to different
                terminal window sizes).
            positions: Relative or absolute positions of log elements
                in each line. If not provided,
                defaults to `[.53, .73, .85, 1.]`.
            print_fn: Print function to use.
                It will be called on each line of the summary.
                You can set it to a custom function
                in order to capture the string summary.
        """
        def print_row(fields, positions_):
            line = ''
            for j in range(len(fields)):
                if j > 0:
                    line = line[:-1] + ' '
                line += str(fields[j])
                line = line[:positions_[j]]
                line += ' ' * (positions_[j] - len(line))
            print_fn(line)

        def print_module_summary(module):
            """Prints a summary for a single module.

            # Arguments
                module: target module.
            """
            called_at = []
            idx = -1
            for _ in range(self._module_stack.count(module)):
                idx = self._module_stack.index(module, idx+1)
                called_at.append(idx)
            fields = [self._getattr(module, 'name') + ' (' + (self._getattr(module, '__class__')).__name__ + ')',
                      module.owner,
                      str((self._getattr(module, 'count_params'))()),
                      str(called_at)]
            print_row(fields, positions)

        self._is_callable()
        line_length = line_length or 98
        positions = positions or [.53, .73, .85, 1.]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ['Module (type)', 'Owner', 'Param #', 'Called at pos.']

        print_fn('\n' + '_' * line_length)
        print_fn('Capsule name: {}'.format(self.name))
        print_fn('Number of prototypes: {:,}'.format(self.proto_number))
        print_fn('Number of capsules: {:,}'.format(self.capsule_number))

        print_fn('=' * line_length)
        print_row(to_display, positions)
        print_fn('=' * line_length)

        modules = self._get_modules()
        for i in range(len(modules)):
            print_module_summary(modules[i])
            if i == len(modules) - 1:
                print_fn('=' * line_length)
            else:
                print_fn('_' * line_length)

        trainable_count = self.count_params()
        non_trainable_count = int(
            np.sum([K.count_params(p) for p in set(self.non_trainable_weights)]))

        print_fn('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print_fn('Trainable params: {:,}'.format(trainable_count))
        print_fn('Non-trainable params: {:,}'.format(non_trainable_count))
        print_fn('_' * line_length + '\n')

    def count_params(self):
        """Count the total number of scalars composing the weights.

        # Returns
            An integer count.
        """
        return int(np.sum([K.count_params(p) for p in set(self.weights)]))


# Todo: test self.input_to_dict
class InputModule(Module):
    """Should handle init of second d input (two-dimensional), reshape to vector size if needed

    return: list of tensor [inputs, d]

    signal_shape: if list or tuple (manually specified shape):
        e.g.: (-1, dim) --> reshape over all: Capsule dimension 1 --> vectors
              (channels, dim1, -1) --> reshape just over channels: capsule dimension 2: --> image
        if int: short cut for (-1, dim)
        if None: bypassing of inputs...no reshape

    if signal is not module:
        - generating of diss and

    Here we have to tile the signal. There is no way around to avoid the storage expensive operation, because we need
    need access to all pre-processed vectors at the routing. In classical layer we can avoid this by formulating the
    operation as one matrix operation.

    The shape of signals is: (batch, num_capsules, channels, dim1, ..., dimN) --> tile to num_capsules

    It's not necessary to tile to proto_number. proto_number is always greater or equal caps_number. Thus if you wanna
    have a prototype wise processing define in a first capsule caps_number as proto_number and make your prototype
    based processing. Then, define a second capsule with proto_number and caps_number and make the remaining processing.

    If you wanna have no tiling, because you wanna make a general processing of the input stream. Define a capsule
    layer with just one capsule and make your processing. Continue with a second capsule and etc.

    shapes of inputs:
        - if module signal is False:
            - inputs = [batch, dim1', ..., dimN']
        - if module signal is True:
            - inputs = [signals, diss]
            - signals = [batch, capsule_number_of_previous, dim1, ..., dimN]
            - diss = [batch, capsule_number_of_previous]
    shapes of the outputs:
        - outputs = [signals, diss]
        - signals = [batch, capsule_number, channels, dim1, ..., dimN]
        - diss = [batch, proto_number, channels]

    If module_input is True:
        - make the tiling of diss and signals
        - make a reshape if signal_shape is not None
        - check that diss fits proto_number
        - overwrite old diss if dissimilarity_initializer is not None or if a dissimilarity_tensor is given
          (check shape)
    is False:
        - make reshape
        - tiling
        - init of diss or routing of tensor
    """
    def __init__(self,
                 signal_shape=None,
                 input_to_dict=False,
                 dissimilarity_initializer=None,
                 dissimilarity_regularizer=None,
                 dissimilarity_constraint='NonNeg',
                 **kwargs):

        if signal_shape is not None:
            if isinstance(signal_shape, (list, tuple)) and len(signal_shape) >= 2:
                self._signal_shape = list(signal_shape)
            elif isinstance(signal_shape, (list, tuple)) and len(signal_shape) == 1 and \
                    isinstance(signal_shape[0], int):
                self._signal_shape = signal_shape[0]
            elif isinstance(signal_shape, int):
                self._signal_shape = signal_shape
            else:
                raise ValueError("'signal_shape' must be list or tuple with len()>=1, int or None.")
        else:
            self._signal_shape = signal_shape

        # read input from list and convert it to dict
        if isinstance(input_to_dict, bool):
            self.input_to_dict = input_to_dict
        else:
            raise TypeError("input_to_dict must be bool. You provide: " + str(input_to_dict))

        self.dissimilarity_initializer = initializers.get(dissimilarity_initializer) \
            if dissimilarity_initializer is not None else None
        self.dissimilarity_regularizer = regularizers.get(dissimilarity_regularizer)
        self.dissimilarity_constraint = constraints.get(dissimilarity_constraint)

        self.dissimilarity = None
        # without batch dim
        self.signal_shape = None

        super(InputModule, self).__init__(module_output=True, **kwargs)

    def _build(self, input_shape):
        if not self.built:
            # check that the input_shape has the correct shape for automatic conversion
            if self.input_to_dict:
                if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
                    raise TypeError("If you provide a module input as list or tuple the length must be two. Otherwise"
                                    "we can't convert the signal automatically to a module input. You provide: "
                                    + str(input_shape))

            if self._module_input or self.input_to_dict:
                signal_shape = shape_inference(input_shape[0][1:], self._signal_shape)
            else:
                signal_shape = shape_inference(input_shape[1:], self._signal_shape)

            # init of dissimilarity
            if self.dissimilarity_initializer is not None:
                self.dissimilarity = self.add_weight(shape=(signal_shape[0],),
                                                     initializer=self.dissimilarity_initializer,
                                                     name='dissimilarity',
                                                     regularizer=self.dissimilarity_regularizer,
                                                     constraint=self.dissimilarity_constraint)
            # routing of dissimilarity
            else:
                # routing not possible
                if not self._module_input and not self.input_to_dict:
                    raise TypeError("If the input is not a module input you have to provide a valid dissimilarity "
                                    "initializer like 'zeros' and not None.")
                # check if diss has the correct channel dimension
                else:
                    if signal_shape[0] != input_shape[1][1]:
                        raise ValueError("The number of channels after reshape of signals must be equal to the number "
                                         "of channels for diss. You provide: diss[1]=" + str(input_shape[1][1]) +
                                         " and signals[1]=" + str(signal_shape[0]))

            # set signal_shape if all tests are passed
            self.signal_shape = signal_shape

            # Set input spec.
            if self._module_input or self.input_to_dict:
                self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                                   InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]
            else:
                self.input_spec = InputSpec(shape=(None,) + tuple(input_shape[1:]))

    def _call(self, inputs, **kwargs):
        if self._module_input or self.input_to_dict:
            signals = inputs[0]
            diss = inputs[1]

        else:
            signals = inputs
            diss = None

        batch_size = None

        with K.name_scope('get_signals'):
            if self._signal_shape is not None:
                batch_size = K.shape(signals)[0] if batch_size is None else batch_size
                signals = K.reshape(signals, (batch_size,) + self.signal_shape)

            signals = K.expand_dims(signals, axis=1)

            signals = K.tile(signals, [1, self.capsule_number] + list(np.ones((len(self.signal_shape)), dtype=int)))

        with K.name_scope('get_dissimilarities'):
            if self.dissimilarity is not None:
                batch_size = K.shape(signals)[0] if batch_size is None else batch_size
                diss = K.expand_dims(K.expand_dims(self.dissimilarity, 0), 0)
                diss = K.tile(diss, [batch_size, self.proto_number, 1])
            else:
                diss = K.expand_dims(diss, 1)
                diss = K.tile(diss, [1, self.proto_number, 1])

        return {0: signals, 1: diss}

    def _compute_output_shape(self, input_shape):
        """Coming soon
        """
        if input_shape is None or not isinstance(input_shape, (list, tuple)) or not len(input_shape) >= 2:
            raise ValueError("'input_shape' must be list or tuple with len()>=2.")

        if not self.built:
            if self._module_input or self.input_to_dict:
                batch_size = input_shape[0][0]
                signal_shape = input_shape[0][2:]
            else:
                batch_size = input_shape[0]
                signal_shape = input_shape[1:]

            signal_shape = shape_inference(signal_shape, self._signal_shape)

            return [(batch_size, self.capsule_number) + signal_shape,
                    (batch_size, self.proto_number, signal_shape[0])]

        else:
            if self._module_input or self.input_to_dict:
                if tuple(self.input_spec[0].shape[1:]) != tuple(input_shape[0][1:]):
                    raise ValueError('Input is incompatible with module ' + self.name + ': expected signal shape='
                                     + str(tuple(self.input_spec[0].shape[1:])) + ', found signal shape='
                                     + str(tuple(input_shape[0][1:])))
                if tuple(self.input_spec[1].shape[1:]) != tuple(input_shape[1][1:]):
                    raise ValueError('Input is incompatible with module ' + self.name + ': expected diss shape='
                                     + str(tuple(self.input_spec[1].shape[1:])) + ', found diss shape='
                                     + str(tuple(input_shape[1][1:])))
                batch_size = input_shape[0][0]
            else:
                if tuple(self.input_spec.shape[1:]) != tuple(input_shape[1:]):
                    raise ValueError('Input is incompatible with module ' + self.name + ': expected input shape='
                                     + str(tuple(self.input_spec.shape[1:])) + ', found input shape='
                                     + str(tuple(input_shape[1:])))
                batch_size = input_shape[0]

            return [(batch_size, self.capsule_number) + self.signal_shape,
                    (batch_size, + self.proto_number, self.signal_shape[0])]

    def get_config(self):
        config = {'signal_shape': self._signal_shape,
                  'input_to_dict': self.input_to_dict,
                  'dissimilarity_initializer': initializers.serialize(self.dissimilarity_initializer),
                  'dissimilarity_regularizer': regularizers.serialize(self.dissimilarity_regularizer),
                  'dissimilarity_constraint': constraints.serialize(self.dissimilarity_constraint),
                  }
        super_config = super(InputModule, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


# Todo: OutputModule test
class OutputModule(Module):
    """push the number of capsules as channels and set capsule_number to one

    shapes of the outputs:
        - outputs = [signals, diss]
        - signals = [batch, capsule_number, channels, dim1, ..., dimN]
        - diss = [batch, proto_number, channels]
    """
    def __init__(self,
                 squeeze_capsule_dim=False,
                 output_to_list=False,
                 **kwargs):

        # needed if you just want to use the capsule as a vector processing unit. Initialize a capsule layer with one
        # capsule and squeeze the dimension at the end.
        if isinstance(squeeze_capsule_dim, bool):
            self.squeeze_capsule_dim = squeeze_capsule_dim
        else:
            raise TypeError("squeeze_capsule_dim must be bool. You provide: " + str(squeeze_capsule_dim))

        # convert the output from dict to list if needed
        if isinstance(output_to_list, bool):
            self.output_to_list = output_to_list
        else:
            raise TypeError("output_to_list must be bool. You provide: " + str(output_to_list))

        super(OutputModule, self).__init__(module_input=True,
                                           module_output=not self.output_to_list,
                                           need_capsule_params=True,
                                           **kwargs)

    def _build(self, input_shape):
        if not self.built:
            if input_shape[0][1] != input_shape[1][1]:
                raise ValueError("The number of capsules must be equal to the number of prototypes. You provide "
                                 + str(input_shape[0][1]) + "!=" + str(input_shape[1][1]))

            if self.squeeze_capsule_dim:
                if input_shape[0][1] != 1:
                    raise ValueError("To squeeze the capsule dimension, teh dimension must be one. You provide: "
                                     + str(input_shape))

            self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                               InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        if self.squeeze_capsule_dim:
            return {0: K.squeeze(inputs[0], 1), 1: K.squeeze(inputs[1], 1)}
        else:
            return inputs

    def _compute_output_shape(self, input_shape):
        if self.squeeze_capsule_dim:
            signals = list(input_shape[0])
            diss = list(input_shape[1])

            del signals[1]
            del diss[1]
            return [tuple(signals), tuple(diss)]
        else:
            return input_shape

    def get_config(self):
        config = {'squeeze_capsule_dim': self.squeeze_capsule_dim,
                  'output_to_list': self.output_to_list}
        super_config = super(OutputModule, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


def shape_inference(input_shape, target_shape):
    """coming soon

    return: signal shape without batch dimension
    """
    if target_shape is not None:
        if isinstance(target_shape, int):
            target_shape = [-1, target_shape]
        else:
            target_shape = target_shape

        if None in input_shape:
            raise ValueError("You have to fully define the input_shape without `None`. You provide: "
                             + str(input_shape))

        if target_shape.count(0) == 0 and np.all(np.array(target_shape) >= -1):
            if target_shape.count(-1) == 1:
                # find shape inference index
                idx = target_shape.index(-1)
                target_shape[idx] = 1

                # shape inference possible?
                if np.prod(input_shape) % np.prod(target_shape) != 0:
                    target_shape[idx] = -1
                    raise ValueError('Cannot reshape tensor of shape ' + str(tuple(input_shape)) +
                                     ' into shape ' + str(tuple(target_shape)) + '.')

                # compute missing dimension
                else:
                    dim = np.prod(input_shape) // np.prod(target_shape)
                    target_shape[idx] = dim

            elif target_shape.count(-1) > 1:
                raise ValueError('Can only infer one unknown dimension. You provide ' + str(tuple(target_shape)))

        else:
            raise ValueError('Cannot reshape to the specified shape: ' + str(tuple(target_shape)))

    else:
        target_shape = list(input_shape)

    # final shape check
    if np.prod(target_shape) != np.prod(input_shape):
        raise ValueError('Cannot reshape a tensor of shape ' + str(tuple(input_shape)) + ' into shape '
                         + str(tuple(target_shape)) + '.')

    return tuple(target_shape)
