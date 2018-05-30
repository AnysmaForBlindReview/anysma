# -*- coding: utf-8 -*-
""" It's an implementation of the traditional GMLVQ.
 Final result is around 97%.
"""

from keras import callbacks
from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers
from keras import Input
from keras.callbacks import EarlyStopping

from anysma import Capsule
from anysma.modules import InputModule
from anysma.modules.routing import SqueezeRouting
from anysma.modules.measuring import OmegaDistance
from anysma.modules.final import Classification
from anysma.callbacks import TensorBoard
from anysma.losses import glvq_loss
from anysma.datasets import tecator

import os

import matplotlib
matplotlib.use('Agg')  # needed to avoid server errors
from matplotlib import pyplot as plt

import numpy as np


batch_size = 3
num_classes = 2
epochs = 100
lr = 0.00001
lr_decay_factor = 0.9995
save_dir = './output'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

x_train, y_train = tecator.load_data()

x_train = x_train.astype('float32')
y_train = to_categorical(y_train, num_classes).astype('float32')

inputs = Input((x_train.shape[1], ))

# define MLVQ prototypes
diss = OmegaDistance(linear_factor=None,
                     squared_dissimilarity=True,
                     matrix_scope='global',
                     matrix_constraint='OmegaNormalization',
                     signal_output='signals')

caps = Capsule(prototype_distribution=(1, num_classes))
caps.add(InputModule(signal_shape=x_train.shape[1], trainable=False, dissimilarity_initializer='zeros'))
caps.add(diss)
caps.add(SqueezeRouting())
caps.add(Classification(probability_transformation='flip', name='lvq_capsule'))

outputs = caps(inputs)

# pre-train the model
pre_train_model = Model(inputs=inputs, outputs=diss.input[0])
diss_input = pre_train_model.predict(x_train, batch_size=batch_size)
diss.pre_training(diss_input, y_train, capsule_inputs_are_equal=True)

model = Model(inputs, outputs[2])
model.compile(loss=glvq_loss, optimizer=optimizers.Adam(lr=lr), metrics={'lvq_capsule': 'accuracy'})

model.summary()

# callbacks
log = callbacks.CSVLogger(os.path.join(save_dir, 'log.csv'))
tb = TensorBoard(log_dir=os.path.join(save_dir, 'tensorboard-logs'),
                 batch_size=batch_size, histogram_freq=True,
                 write_grads=True,
                 write_images=True)

es = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')

model.fit(x_train,
          y_train,
          validation_data=[x_train, y_train],
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          verbose=1,
          callbacks=[log, tb, es])

model.save_weights(os.path.join(save_dir, 'trained_model.h5'))

omega = diss.get_weights()[1]
plt.imshow(np.matmul(omega, omega.transpose()))
plt.colorbar()
plt.title('Classification Correlation Matrix')
plt.savefig(os.path.join(save_dir, 'CCM_Tecator.png'))
