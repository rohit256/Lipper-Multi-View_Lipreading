from keras.utils import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D, LSTM
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.layers import merge, Input
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.layers.wrappers import *
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
import numpy as np
import os
from keras.regularizers import *
from keras.layers import LSTM, Input
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.layers import Dense, Activation, Flatten,Concatenate,concatenate
from keras.models import Model
from sklearn.utils import shuffle
import keras
import keras.backend as BK
import sys
from os.path import isfile, join
import shutil
import h5py
import os.path
import glob
import audio_tools as aud
import moduletest

FRAME_ROWS = 128
FRAME_COLS = 128
SR = moduletest.SamplingRate
LPC_ORDER = moduletest.LPCOrder
NFRAMES = moduletest.NumberOfFrames
MARGIN = int(NFRAMES/2)
COLORS = 1
CHANNELS = NFRAMES
TRAIN_PER = (len(moduletest.SpeakerTrain)*1.0)/(len(moduletest.SpeakerTrain)+len(moduletest.SpeakerTest))
LR = 0.005
nb_pool = 2
BATCH_SIZE = 10
DROPOUT = 0 
DROPOUT2 = 0.0
EPOCHS = 50
FINETUNE_EPOCHS = 10
activation_func2 = 'tanh'
net_out = 50
reg = 0.0005

respath = '../results/speaker_independent/1_View/'
weight_path = join(respath,'weights/')
datapath = '../../lipsync/dataset/numpy_datasets/'
#datapath = '../dataset/numpy_datasets/'
def savedata(Ytr, Ytr_pred, Yte, Yte_pred,v1):
    respath_view = join(respath,'View='+str(v1)+'/')
    if not os.path.exists(respath_view):
        os.makedirs(respath_view)
    speakerlist = '('+str(moduletest.SpeakerTrain)[1:-1] + '||||' + str(moduletest.SpeakerTest)[1:-1]+')'
    nameoffile = 'STCNN'+'_'+ speakerlist +'_'+str(v1)+'_'+str(SR)+'_'+str(NFRAMES)
    np.save(join(respath_view,'Ytr_'+nameoffile+'.npy'),Ytr)
    np.save(join(respath_view,'Ytr_pred_'+nameoffile+'.npy'),Ytr_pred)
    np.save(join(respath_view,'Yte_'+nameoffile+'.npy'),Yte)
    np.save(join(respath_view,'Yte_pred_'+nameoffile+'.npy'),Yte_pred)

def standardize_data(Xtr, Ytr, Xte, Yte):
        Xtr = Xtr.astype('float32')
        Xte = Xte.astype('float32')
        Xtr /= 255
        Xte /= 255
        xtrain_mean = np.mean(Xtr)
        Xtr = Xtr-xtrain_mean
        Xte = Xte-xtrain_mean
        Y_means = np.mean(Ytr,axis=0) 
        Y_stds = np.std(Ytr, axis=0)
        Ytr_norm = ((Ytr-Y_means)/Y_stds)
        Yte_norm = ((Yte-Y_means)/Y_stds)
        return Xtr, Ytr_norm, Xte, Yte_norm, Y_means, Y_stds

def load_data(datapath,view):
	finalviddata = []
	finalauddata = []
	for i in np.concatenate((moduletest.SpeakerTrain,moduletest.SpeakerTest)):
		speaker = i
		viddata_path = join(datapath,'viddata_'+str(SR)+'_'+str(NFRAMES)+'_'+str(speaker)+'_'+str(view)+'.npy')
		auddata_path = join(datapath,'auddata_'+str(SR)+'_'+str(NFRAMES)+'_'+str(speaker)+'.npy')
		if isfile(viddata_path) and isfile(auddata_path):
			print ('Loading data...')
			viddata = np.load(viddata_path)
			auddata = np.load(auddata_path)
			if(len(finalviddata)==0):
				finalviddata = viddata
			else :
				finalviddata = np.concatenate((finalviddata,viddata),axis=0)
			if(len(finalauddata)==0):
				finalauddata = auddata
			else :
				finalauddata = np.concatenate((finalauddata,auddata),axis=0)
		else:
			print ('Preprocessed data not found.')
			return None, None
	print ('Done.')
	#print finalviddata.shape,finalauddata.shape
	return finalviddata, finalauddata

def split_data(viddata, auddata):
    vidctr = len(auddata)
    Xtr = viddata[:int(vidctr*TRAIN_PER),:,:,:]
    Ytr = auddata[:int(vidctr*TRAIN_PER),:]
    Xte = viddata[int(vidctr*TRAIN_PER):,:,:,:]
    Yte = auddata[int(vidctr*TRAIN_PER):,:]
    return (Xtr, Ytr), (Xte, Yte)


def build_model(net_out):
	model = Sequential()

	model.add(Conv2D(32, kernel_size=( 3, 3), init='he_normal', input_shape=(CHANNELS, FRAME_ROWS, FRAME_COLS),
		            data_format='channels_first', padding='same', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization())
	model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), data_format='channels_first'))
	#model.add(Dropout(DROPOUT))

	model.add(Conv2D(32, 3, init='he_normal',data_format='channels_first', padding='same',kernel_regularizer=l2(reg)))
	model.add(BatchNormalization())
	model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), data_format='channels_first'))
	#model.add(Dropout(DROPOUT))

	model.add(Conv2D(32, 3, init='he_normal',data_format='channels_first', padding='same', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization())
	model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), data_format='channels_first'))
	model.add(Dropout(DROPOUT))

	model.add(Conv2D(64, 3, init='he_normal',data_format='channels_first', padding='same', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization())
	model.add(LeakyReLU())
	#model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), data_format='channels_first'))
	#model.add(Dropout(DROPOUT2))

	model.add(Conv2D(64, 3, init='he_normal',data_format='channels_first', padding='same', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization())
	model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), data_format='channels_first'))
	model.add(Dropout(DROPOUT2))

	model.add(Conv2D(128, 3, init='he_normal',data_format='channels_first', padding='same', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization())
	model.add(LeakyReLU())
	#model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), data_format='channels_first'))
	#model.add(Dropout(DROPOUT2))

	model.add(Conv2D(128, 3, init='he_normal',data_format='channels_first', padding='same', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization())
	model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), data_format='channels_first'))
	model.add(Dropout(DROPOUT2))

	model.add(Flatten())

	model.add(Dense(128, init='he_normal'))
	model.add(BatchNormalization())
	model.add(Dense(net_out))

	return model


def corr2_mse_loss(a,b):
	a = BK.tf.subtract(a, BK.tf.reduce_mean(a))
	b = BK.tf.subtract(b, BK.tf.reduce_mean(b))
	tmp1 = BK.tf.reduce_sum(BK.tf.multiply(a,a))
	tmp2 = BK.tf.reduce_sum(BK.tf.multiply(b,b))
	tmp3 = BK.tf.sqrt(BK.tf.multiply(tmp1,tmp2))
	tmp4 = BK.tf.reduce_sum(BK.tf.multiply(a,b))
	r = -BK.tf.divide(tmp4,tmp3)
	m=BK.tf.reduce_mean(BK.tf.square(BK.tf.subtract(a, b)))
	rm=BK.tf.add(r,m)
	return rm

def train_net(model, Xtr, Ytr_norm, Xte, Yte_norm, batch_size=BATCH_SIZE, epochs=EPOCHS, finetune=False):
	if finetune:
		lr = LR/10
	else:
		lr = LR
	adam = Adam(lr=lr)
	model.compile(loss='mse', optimizer=adam)
	history = model.fit(Xtr, Ytr_norm, batch_size=batch_size, nb_epoch=epochs,
				verbose=1, validation_data=(Xte, Yte_norm))
	return model

def predict(model, X, Y_means, Y_stds, batch_size=BATCH_SIZE):
	Y_pred = model.predict(X, batch_size=batch_size, verbose=1)
	Y_pred = (Y_pred*Y_stds+Y_means)
	return Y_pred

def main():
	for view in range(1,6):
		#view = 1
		viddata, auddata = load_data(datapath,view) 
		(Xtr,Ytr), (Xte, Yte) = split_data(viddata, auddata)
		print (Xtr.shape, Ytr.shape)
		print (Xte.shape, Yte.shape)
		Xtr, Ytr_norm, Xte, Yte_norm, Y_means, Y_stds = standardize_data(Xtr, Ytr, Xte, Yte)
		model = build_model(net_out)
		model = train_net(model, Xtr, Ytr_norm, Xte, Yte_norm)
		model = train_net(model, Xtr, Ytr_norm, Xte, Yte_norm, epochs=FINETUNE_EPOCHS, finetune=True)
		Ytr_pred = predict(model, Xtr, Y_means, Y_stds)
		Yte_pred = predict(model, Xte, Y_means, Y_stds)
		savedata(Ytr, Ytr_pred, Yte, Yte_pred, view)


if __name__ == "__main__":
    main()
