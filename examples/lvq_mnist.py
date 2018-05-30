# -*- coding: utf-8 -*-
""" Traditional GLVQ.
Final result is around 90%.
"""

from keras import callbacks
from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers
from keras import Input
from keras.callbacks import EarlyStopping
from keras.datasets import mnist

from anysma import Capsule
from anysma.modules import InputModule
from anysma.modules.routing import SqueezeRouting
from anysma.modules.measuring import MinkowskiDistance
from anysma.modules.competition import NearestCompetition
from anysma.modules.final import Classification
from anysma.callbacks import TensorBoard
from anysma.losses import glvq_loss

import os

import matplotlib
matplotlib.use('Agg')  # needed to avoid cloud errors
from matplotlib import pyplot as plt

import numpy as np


batch_size = 32
num_classes = 10
protos_per_class = 3
epochs = 25
lr = 0.0001
lr_decay_factor = 0.9995
save_dir = './output'

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train.astype('float32'))
y_test = to_categorical(y_test.astype('float32'))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

inputs = Input(x_train.shape[1:])

# define MLVQ prototypes
diss = MinkowskiDistance(linear_factor=None,
                         squared_dissimilarity=True,
                         signal_output='signals')

caps = Capsule(prototype_distribution=(protos_per_class, num_classes))
caps.add(InputModule(signal_shape=(1,) + x_train.shape[1:], trainable=False, dissimilarity_initializer='zeros'))
caps.add(diss)
caps.add(SqueezeRouting())
caps.add(NearestCompetition(use_for_loop=False))
caps.add(Classification(probability_transformation='flip', name='lvq_capsule'))

outputs = caps(inputs)

# pre-train the model over 10000 random digits
idx = np.random.randint(0, len(x_train) - 1, (min(10000, len(x_train)),))
pre_train_model = Model(inputs=inputs, outputs=diss.input[0])
diss_input = pre_train_model.predict(x_train[idx, :], batch_size=batch_size)
diss.pre_training(diss_input, y_train[idx], capsule_inputs_are_equal=True)

model = Model(inputs, outputs[2])
model.compile(loss=glvq_loss, optimizer=optimizers.Adam(lr=lr), metrics={'lvq_capsule': 'accuracy'})

model.summary()

# callbacks
log = callbacks.CSVLogger(os.path.join(save_dir, 'log.csv'))
tb = TensorBoard(log_dir=os.path.join(save_dir, 'tensorboard-logs'),
                 batch_size=batch_size)

es = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')

model.fit(x_train,
          y_train,
          validation_data=[x_test, y_test],
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          verbose=1,
          callbacks=[log, tb, es])

model.save_weights(os.path.join(save_dir, 'trained_model.h5'))

protos = diss.get_weights()[0]
for i in range(num_classes):
    for j in range(protos_per_class):
        plt.subplot(protos_per_class, num_classes, j * num_classes + i + 1)
        img = plt.imshow(protos[i*protos_per_class + j, :, :])
        img.set_cmap('gray')
        plt.axis('off')
plt.savefig(os.path.join(save_dir, 'protos.png'), bbox_inches='tight')

