from __future__ import absolute_import

from .transformations import *
from .measuring import *
from .routing import *
from .competition import *
from .final import *
from ..capsule import InputModule, OutputModule, Module  # make the modules visible


def globals_modules():
    globs = globals()
    from ..capsule import Capsule
    globs.update({'Capsule': Capsule})
    return globs

# not sure if we really need this
# from keras.layers import serialize as keras_serialize
# from keras.utils import deserialize_keras_object
# def serialize(module):
#     return keras_serialize(module)
#
#
# def deserialize(config, custom_objects=None):
#     globs = globals_modules  # All modules.
#     return deserialize_keras_object(config,
#                                     module_objects=globs,
#                                     custom_objects=custom_objects,
#                                     printable_module_name='module')
