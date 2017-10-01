import keras
import tensorflow as tf
from keras.layers import Conv2D
#import keras.activations as Activation
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.core import Dropout

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input



class Base_Vgg16(img_input):
	def __init__(self):
		self.lr = 0.001
		self.weight_filter = keras.initializers.RandomNormal(mean=0.0, stddev=0.001)
	def model(self, input_shape)
		net = {}
		# input block
		input_tensor = Input(shape=input_shape)
		net['input'] = input_tensor
		# block 1
		'''
		input size: (1,3,448,448)
		output size: (1,64,224,224)
		'''
		net['conv1_1'] = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(net['input'])
		net['conv1_2'] = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(net['conv1_1'])
		net['pool1'] = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(net['conv1_2'])

		# block 2
		''' output size: (1,128,112,112)'''
		net['conv2_1'] = Conv2D(128, (3,3), activation='relu', padding='same', name='conv2_1')(net['pool1'])
		net['conv2_2'] = Conv2D(128, (3,3), activation='relu', padding='same', name='conv2_2')(net['conv2_1'])
		net['pool2'] = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(net['conv2_2'])

		# block 3
		''' output size: (1,256,112,112)'''
		net['conv3_1'] = Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_1')(net['pool2'])
		net['conv3_2'] = Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_2')(net['conv3_1'])
		net['conv3_3'] = Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_3')(net['conv3_2'])
		net['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(net['conv3_3'])

		# block 4
		''' 
		output size: (1,512,56,56)
		original pool4 is removed (or use a element wise pool)
		'''
		net['conv4_1'] = Conv2D(512, (3,3), activation='relu', padding='same', name='conv4_1')(net['pool3'])
		net['conv4_2'] = Conv2D(512, (3,3), activation='relu', padding='same', name='conv4_2')(net['conv4_1'])
		net['conv4_3'] = Conv2D(512, (3,3), activation='relu', padding='same', name='conv4_3')(net['conv4_2']) 

		# block 5
		''' 
		output size: (1,512,56,56)
		original pool5 is removed (or use a element wise pool)
		'''
		net['conv5_1'] = Conv2D(512, (3,3), activation='relu', padding='same', name='conv5_1')(net['conv4_3'])
		net['conv5_2'] = Conv2D(512, (3,3), activation='relu', padding='same', name='conv5_2')(net['conv5_1'])
		net['conv5_3'] = Conv2D(512, (3,3), activation='relu', padding='same', name='conv5_3')(net['conv5_2'])

		# block 6 - FCN BN and Dropout
		
		net['fcn6'] = Conv2D(1024, (7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=self.weight_filter,name='fcn6')(net['conv5_3'])
		net['fconv6_BN'] = BatchNormalization(center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones')(net['fcn6'])
		net['fconv6_BN'] = Activation('relu')(net['fconv6_BN'])
		net['fconv6_BN'] = Dropout(rate=0.5)(net['fconv6_BN'])

		# generate the model
		model = Model(net['input'], net['fcn6_BN'])

		return net, model
