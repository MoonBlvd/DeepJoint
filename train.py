import numpy as np
import tensorflow as tf
import keras
from keras.models import Model

from base_vgg import BaseVgg16
from structural_layers import StructuralLayers
import loss

# learning rate scheduler, decrease the learning rate gradually based on the epoch.
def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

# build the layers
input_shape = (448,448,3)
base = BaseVgg16()
structural = StructuralLayers()

net = base.model(input_shape)
net = structural.add_direction_A_layers(net)
net = structural.add_direction_B_layers(net)
net = structural.concate_branches(net)
net = structural.fconv9(net)

# generate the model
model = Model(net['input'], net['fconv9'])
'''should load the model weight here'''
print (model.layers)

# compile model
base_lr = 3e-4
optim = keras.optimizers.Adam(lr=base_lr)

model.compile(optimizer=optim, loss=loss.softmax_loss)

# add callbacks
callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True),
             keras.callbacks.LearningRateScheduler(schedule)]
print ("The callbacks config:", callbacks)

# train the model
num_epoch = 20
history = model.fit(self, x=None, y=None, 
                    batch_size=None, epochs=num_epoch, verbose=1, 
                    callbacks=callbacks, 
                    validation_split=0.0, validation_data=None, 
                    shuffle=True, class_weight=None, sample_weight=None, 
                    initial_epoch=0, steps_per_epoch=None, 
                    validation_steps=None)