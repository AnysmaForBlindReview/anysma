# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.callbacks import Callback
from keras.callbacks import TensorBoard as KerasTensorBoard
from keras import backend as K

import warnings
import os


class TensorBoard(KerasTensorBoard):
    """TensorBoard basic visualizations.

    It's a slight modified version (see `if hasattr(layer, 'output')` in set_model()) of the Keras TensorBoard callback:

                                https://github.com/keras-team/keras/blob/master/keras/callbacks.py

    to support list and dict outputs at histograms.
    The main implementation is based on the Keras callback. For the future, we should open a pull request at Keras.

    Additionally, we support `max_outputs` for `tf.summary.image()`.

    [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```sh
    tensorboard --logdir=/full_path_to_your_logs
    ```

    When using a backend other than TensorFlow, TensorBoard will still work
    (if you have TensorFlow installed), but the only feature available will
    be the display of the losses and metrics plots.

    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 images_max_outputs=3,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(TensorBoard, self).__init__(log_dir=log_dir,
                                          histogram_freq=histogram_freq,
                                          batch_size=batch_size,
                                          write_graph=write_graph,
                                          write_grads=write_grads,
                                          write_images=write_images,
                                          embeddings_freq=embeddings_freq,
                                          embeddings_layer_names=embeddings_layer_names,
                                          embeddings_metadata=embeddings_metadata)
        self.images_max_outputs = images_max_outputs

        global tf
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError('You need the TensorFlow module installed to use TensorBoard.')

    def set_model(self, model):
        histogram_freq = self.histogram_freq
        self.histogram_freq = False

        super(TensorBoard, self).set_model(model)

        self.histogram_freq = histogram_freq

        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf.summary.histogram(mapped_weight_name, weight)
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss, weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'
                        grads = [grad.values if is_indexed_slices(grad) else grad for grad in grads]
                        tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1, shape[0], shape[1], 1])
                        elif len(shape) == 3:  # convnet case
                            if K.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0], shape[1], shape[2], 1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1, shape[0], 1, 1])
                        else:
                            # not possible to handle 3D convnets etc.
                            continue

                        shape = K.int_shape(w_img)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf.summary.image(mapped_weight_name, w_img, max_outputs=self.images_max_outputs)

                if hasattr(layer, 'output'):
                    if isinstance(layer.output, dict):
                        for i in layer.output:
                            tf.summary.histogram('{}_out_{}'.format(layer.name, str(i)), layer.output[i])
                    elif isinstance(layer.output, list):
                        for i, o in enumerate(layer.output):
                            tf.summary.histogram('{}_out_{}'.format(layer.name, i), o)
                    else:
                        tf.summary.histogram('{}_out'.format(layer.name), layer.output)

        self.merged = tf.summary.merge_all()

