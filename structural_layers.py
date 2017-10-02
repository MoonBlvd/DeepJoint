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

class StructuralLayers():
	def __init__(self):
		self.lr = 0.0001
		
	def add_direction_A_layers(self, net):
		# ----------------------------------------- Structure 3-2-1 ---------------------------------------

		# fconv7_3, padding should be 3
		weight_filter = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
		net['fconv7_3']=Conv2D(128, (1,1), activation='relu', padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_3')(net['fconv6_BN'])
		net['fconv7_3to2_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_3to2_step1')(net['fconv7_3'])
		net['fconv7_3to2_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_3to2_step2')(net['fconv7_3to2_step1'])

		# fconv7_2, padding should be 3
		net['fconv7_2']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_2')(net['fconv6_BN'])
		net['fconv7_2+']=Add()([net['fconv7_3to2_step2'],net['fconv7_2']]) # element-wise sum
		net['fconv7_2+']=Activation('relu')(net['fconv7_2+'])
		net['fconv7_2to1_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_2to1_step1')(net['fconv7_2+'])
		net['fconv7_2to1_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_2to1_step2')(net['fconv7_2to1_step1'])

		# ----------------------------------------- Structure 6-5-4 ---------------------------------------

		# fconv7_6, padding should be 3
		weight_filter = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
		net['fconv7_6']=Conv2D(128, (1,1), activation='relu', padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_6')(net['fconv6_BN'])
		net['fconv7_6to5_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_6to5_step1')(net['fconv7_6'])
		net['fconv7_6to5_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_6to5_step2')(net['fconv7_6to5_step1'])

		# fconv7_5, padding should be 3
		net['fconv7_5']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_5')(net['fconv6_BN'])
		net['fconv7_5+']=Add()([net['fconv7_6to5_step2'],net['fconv7_5']]) # element-wise sum
		net['fconv7_5+']=Activation('relu')(net['fconv7_5+'])
		net['fconv7_5to4_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_5to4_step1')(net['fconv7_5+'])
		net['fconv7_5to4_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_5to4_step2')(net['fconv7_5to4_step1'])

		print (net['fconv7_2to1_step2'])
		print (net['fconv7_5to4_step2'])

		# ----------------------------------------- Structure 9-8-7-1 ------------------------------------------------

		# fconv7_9, padding should be 3
		weight_filter = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
		net['fconv7_9']=Conv2D(128, (1,1), activation='relu', padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_9')(net['fconv6_BN'])
		net['fconv7_9to8_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_9to8_step1')(net['fconv7_9'])
		net['fconv7_9to8_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_9to8_step2')(net['fconv7_9to8_step1'])

		# fconv7_8, padding should be 3
		net['fconv7_8']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_8')(net['fconv6_BN'])
		net['fconv7_8+']=Add()([net['fconv7_14to13_step2'],net['fconv7_8']]) # element-wise sum
		net['fconv7_8+']=Activation('relu')(net['fconv7_8+'])
		net['fconv7_8to7_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_8to7_step1')(net['fconv7_8+'])
		net['fconv7_8to7_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_8to7_step2')(net['fconv7_8to7_step1'])

		# fconv7_7, padding should be 3
		net['fconv7_7']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_7')(net['fconv6_BN'])
		net['fconv7_7+']=Add()([net['fconv7_8to7_step2'],net['fconv7_7']]) # element-wise sum
		net['fconv7_7+']=Activation('relu')(net['fconv7_7+'])
		net['fconv7_7to1_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_7to1_step1')(net['fconv7_7+'])
		net['fconv7_7to1_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_7to1_step2')(net['fconv7_7to1_step1'])

		# ----------------------------------------- Structure 12-11-10-4 ------------------------------------------------

		# fconv7_12, padding should be 3
		weight_filter = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
		net['fconv7_12']=Conv2D(128, (1,1), activation='relu', padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_12')(net['fconv6_BN'])
		net['fconv7_12to11_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_12to11_step1')(net['fconv7_12'])
		net['fconv7_12to11_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_12to11_step2')(net['fconv7_12to11_step1'])

		# fconv7_11, padding should be 3
		net['fconv7_11']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_11')(net['fconv6_BN'])
		net['fconv7_11+']=Add()([net['fconv7_12to11_step2'],net['fconv7_11']]) # element-wise sum
		net['fconv7_11+']=Activation('relu')(net['fconv7_11+'])
		net['fconv7_11to10_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_11to10_step1')(net['fconv7_11+'])
		net['fconv7_11to10_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_8to7_step2')(net['fconv7_8to7_step1'])

		# fconv7_10, padding should be 3
		net['fconv7_10']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_7')(net['fconv6_BN'])
		net['fconv7_10+']=Add()([net['fconv7_11to10_step2'],net['fconv7_10']]) # element-wise sum
		net['fconv7_10+']=Activation('relu')(net['fconv7_10+'])
		net['fconv7_10to4_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_10to4_step1')(net['fconv7_10+'])
		net['fconv7_10to4_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_10to4_step2')(net['fconv7_10to4_step1'])

		print (net['fconv7_10to4_step2'])

		# ---------------------------- Structure 2(elbow) and 7(hip) to 1(shoulder) --------------------------------
		# fconv7_1, padding should be 3
		net['fconv7_1']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_1')(net['fconv6_BN'])
		net['fconv7_1+']=Add()([net['fconv7_2to1_step2'], net['fconv7_7to1_step2'], net['fconv7_1']]) # element-wise sum
		net['fconv7_1+']=Activation('relu')(net['fconv7_1+'])
		net['fconv7_1to14_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_1to14_step1')(net['fconv7_1+'])
		net['fconv7_1to14_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_1to14_step2')(net['fconv7_1to14_step1'])

		print (net['fconv7_1to14_step2'])

		# ---------------------------- Structure 5(elbow) and 10(hip) to 4(shoulder) -------------------------------
		# fconv7_4, padding should be 3
		net['fconv7_4']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_4')(net['fconv6_BN'])
		net['fconv7_4+']=Add()([net['fconv7_5to4_step2'], net['fconv7_10to4_step2'], net['fconv7_4']]) # element-wise sum
		net['fconv7_4+']=Activation('relu')(net['fconv7_4+'])
		net['fconv7_4to14_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_4to14_step1')(net['fconv7_4+'])
		net['fconv7_4to14_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_4to14_step2')(net['fconv7_4to14_step1'])

		print (net['fconv7_4to14_step2'])

		# ---------------------------- Structure 4(shoulder) and 1(shoulder) to 14(neck) -------------------------------
		# fconv7_14, padding should be 3
		net['fconv7_14']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_14')(net['fconv6_BN'])
		net['fconv7_14+']=Add()([net['fconv7_1to14_step2'], net['fconv7_4to14_step2'], net['fconv7_14']]) # element-wise sum
		net['fconv7_14+']=Activation('relu')(net['fconv7_14+'])
		net['fconv7_14to13_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_14to13_step1')(net['fconv7_14+'])
		net['fconv7_14to13_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_14to13_step2')(net['fconv7_14to13_step1'])

		print (net['fconv7_14to13_step2'])

		# -------------------------------------- Structure 14(neck) to 13(head top) -------------------------------
		# fconv7_13, padding should be 3
		net['fconv7_13']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_13')(net['fconv6_BN'])
		net['pass_14to13']=Add()([net['fconv7_14to13_step2'], net['fconv7_13']]) # element-wise sum
		net['fconv7_13+']=Activation('relu')(net['pass_14to13'])

		print (net['fconv7_13+'])

		return net

	def add_direction_B_layers(self, net):
		# --------------------------------- reverse of the structural tree ------------------------------------

		# -------------------------------------- Structure 13(head top) to 14(neck) -------------------------------
		net['fconvB7_13']=Conv2D(128, (1,1), activation='relu', padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_13')(net['fconv6_BN'])
		net['fconv7_13to14_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_13to14_step1')(net['fconvB7_13'])
		net['fconv7_13to14_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_13to14_step2')(net['fconv7_13to14_step1'])

		# ---------------------------- Structure 14(neck) to 1(shoulder) and 4(shoulder) --------------------------
		net['fconvB7_14']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_14')(net['fconv6_BN'])
		net['fconvB7_14+']=Add()([net['fconv7_13to14_step2'], net['fconvB7_14']]) # element-wise sum
		net['fconvB7_14+']=Activation('relu')(net['fconvB7_14+'])
		net['fconv7_14to1_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_14to1_step1')(net['fconvB7_14+'])
		net['fconv7_14to1_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_14to1_step2')(net['fconv7_14to1_step1'])
		net['fconv7_14to4_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_14to4_step1')(net['fconvB7_14+'])
		net['fconv7_14to4_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_14to4_step2')(net['fconv7_14to4_step1'])

		# ---------------------------- Structure 1(shoulder) to 2(elbow) and 7(hip)
		net['fconvB7_1']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_1')(net['fconv6_BN'])
		net['fconvB7_1+']=Add()([net['fconv7_14to1_step2'], net['fconvB7_1']]) # element-wise sum
		net['fconvB7_1+']=Activation('relu')(net['fconvB7_1+'])
		net['fconv7_1to2_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_1to2_step1')(net['fconvB7_1+'])
		net['fconv7_1to2_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_1to2_step2')(net['fconv7_1to2_step1'])
		net['fconv7_1to7_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_1to7_step1')(net['fconvB7_1+'])
		net['fconv7_1to7_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_1to7_step2')(net['fconv7_1to7_step1'])

		# ---------------------------- Structure 4(shoulder) to 5(elbow) and 10(hip)
		net['fconvB7_4']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_4')(net['fconv6_BN'])
		net['fconvB7_4+']=Add()([net['fconv7_14to4_step2'], net['fconvB7_4']]) # element-wise sum
		net['fconvB7_4+']=Activation('relu')(net['fconvB7_4+'])
		net['fconv7_4to5_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_4to5_step1')(net['fconvB7_4+'])
		net['fconv7_4to5_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_4to5_step2')(net['fconv7_4to5_step1'])
		net['fconv7_4to10_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_4to10_step1')(net['fconvB7_4+'])
		net['fconv7_4to10_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_4to10_step2')(net['fconv7_4to10_step1'])

		# ------------------------------------ Structure 2(elbow) to 3(hand) -----------------------------------------
		net['fconvB7_2']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_2')(net['fconv6_BN'])
		net['fconvB7_2+']=Add()([net['fconv7_1to2_step2'], net['fconvB7_2']]) # element-wise sum
		net['fconvB7_2+']=Activation('relu')(net['fconvB7_2+'])
		net['fconv7_2to3_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_2to3_step1')(net['fconvB7_2+'])
		net['fconv7_2to3_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_2to3_step2')(net['fconv7_2to3_step1'])

		net['fconvB7_3']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_3')(net['fconv6_BN'])
		net['fconvB7_3+']=Activation('relu')(Add()([net['fconv7_2to3_step2'], net['fconv7_3']]))

		# ------------------------------------ Structure 5(elbow) to 6(hand) -----------------------------------------
		net['fconvB7_5']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_5')(net['fconv6_BN'])
		net['fconvB7_5+']=Add()([net['fconv7_4to5_step2'], net['fconvB7_5']]) # element-wise sum
		net['fconvB7_5+']=Activation('relu')(net['fconvB7_5+'])
		net['fconv7_5to6_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_5to6_step1')(net['fconvB7_5+'])
		net['fconv7_5to6_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_5to6_step2')(net['fconv7_5to6_step1'])

		net['fconvB7_6']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_6')(net['fconv6_BN'])
		net['fconvB7_6+']=Activation('relu')(Add()([net['fconv7_5to6_step2'], net['fconv7_6']]))

		# ------------------------------------ Structure 7(hip)-8(knee)-9(foot) -----------------------------------------
		net['fconvB7_7']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_7')(net['fconv6_BN'])
		net['fconvB7_7+']=Add()([net['fconv7_1to7_step2'], net['fconvB7_7']]) # element-wise sum
		net['fconvB7_7+']=Activation('relu')(net['fconvB7_7+'])
		net['fconv7_7to8_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_7to8_step1')(net['fconvB7_7+'])
		net['fconv7_7to8_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_7to8_step2')(net['fconv7_7to8_step1'])

		net['fconvB7_8']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_8')(net['fconv6_BN'])
		net['fconvB7_8+']=Add()([net['fconv7_7to8_step2'], net['fconvB7_8']]) # element-wise sum
		net['fconvB7_8+']=Activation('relu')(net['fconvB7_8+'])
		net['fconv7_8to9_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_8to9_step1')(net['fconvB7_8+'])
		net['fconv7_8to9_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_8to9_step2')(net['fconv7_8to9_step1'])

		net['fconvB7_9']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_9')(net['fconv6_BN'])
		net['fconvB7_9+']=Activation('relu')(Add()([net['fconv7_8to9_step2'], net['fconv7_9']]))

		# ------------------------------------ Structure 10(hip)-11(knee)-12(foot) -----------------------------------------
		net['fconvB7_10']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_10')(net['fconv6_BN'])
		net['fconvB7_10+']=Add()([net['fconv7_4to10_step2'], net['fconvB7_10']]) # element-wise sum
		net['fconvB7_10+']=Activation('relu')(net['fconvB7_10+'])
		net['fconv7_10to11_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_10to11_step1')(net['fconvB7_10+'])
		net['fconv7_10to11_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_10to11_step2')(net['fconv7_10to11_step1'])

		net['fconvB7_11']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_11')(net['fconv6_BN'])
		net['fconvB7_11+']=Add()([net['fconv7_10to11_step2'], net['fconvB7_11']]) # element-wise sum
		net['fconvB7_11+']=Activation('relu')(net['fconvB7_11+'])
		net['fconv7_11to12_step1']=Conv2D(64,(7,7), activation='relu', padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_11to12_step1')(net['fconvB7_11+'])
		net['fconv7_11to12_step2']=Conv2D(128,(7,7), padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconv7_11to12_step2')(net['fconv7_11to12_step1'])

		net['fconvB7_12']=Conv2D(128, (1,1), padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_12')(net['fconv6_BN'])
		net['fconvB7_12+']=Activation('relu')(Add()([net['fconv7_11to12_step2'], net['fconvB7_12']]))

		print (net['fconvB7_3+'])
		print (net['fconvB7_6+'])
		print (net['fconvB7_9+'])
		print (net['fconvB7_12+'])

		net['fconvB7_27']=Conv2D(512, (1,1), activation='relu', padding='valid',use_bias=True, bias_initializer='zeros', kernel_initializer=weight_filter,name='fconvB7_27')(net['fconv6_BN'])
		net['fconvB7_27+'] = Dropout(rate=0.5)(net['fconvB7_27'])

		return net

	def concate_branches(self, net):
		#-------------------------------------------- concate branches -------------------------------------------

		net['fconv7_1_concat']=Concatenate(axis=3)([net['fconv7_1+'], net['fconvB7_1+']])
		net['fconv7_2_concat']=Concatenate(axis=3)([net['fconv7_2+'], net['fconvB7_2+']])
		net['fconv7_3_concat']=Concatenate(axis=3)([net['fconv7_3'], net['fconvB7_3+']])
		net['fconv7_4_concat']=Concatenate(axis=3)([net['fconv7_4+'], net['fconvB7_4+']])
		net['fconv7_5_concat']=Concatenate(axis=3)([net['fconv7_5+'], net['fconvB7_5+']])
		net['fconv7_6_concat']=Concatenate(axis=3)([net['fconv7_6'], net['fconvB7_6+']])
		net['fconv7_7_concat']=Concatenate(axis=3)([net['fconv7_7+'], net['fconvB7_7+']])
		net['fconv7_8_concat']=Concatenate(axis=3)([net['fconv7_8+'], net['fconvB7_8+']])
		net['fconv7_9_concat']=Concatenate(axis=3)([net['fconv7_9'], net['fconvB7_9+']])
		net['fconv7_10_concat']=Concatenate(axis=3)([net['fconv7_10+'], net['fconvB7_10+']])
		net['fconv7_11_concat']=Concatenate(axis=3)([net['fconv7_11+'], net['fconvB7_11+']])
		net['fconv7_12_concat']=Concatenate(axis=3)([net['fconv7_12'], net['fconvB7_12+']])
		net['fconv7_13_concat']=Concatenate(axis=3)([net['fconv7_13+'], net['fconvB7_13']])
		net['fconv7_14_concat']=Concatenate(axis=3)([net['fconv7_14+'], net['fconvB7_14+']])
		print (net['fconv7_14_concat'])