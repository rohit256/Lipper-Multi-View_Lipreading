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
BATCH_SIZE = 26
DROPOUT = 0 
DROPOUT2 = 0.0
EPOCHS = 30
FINETUNE_EPOCHS = 5
activation_func2 = 'tanh'
net_out = 50
reg = 0.0005

respath = '../results/speaker_independent/2_View/'
weight_path = join(respath,'weights/')
datapath = '../../lipsync/dataset/numpy_datasets/'

def savedata(Ytr, Ytr_pred, Yte, Yte_pred,v1,v2):
    respath_view = join(respath,'View='+str(v1)+','+str(v2)+'/')
    if not os.path.exists(respath_view):
        os.makedirs(respath_view)
    speakerlist = '('+str(moduletest.SpeakerTrain)[1:-1] + '||||' + str(moduletest.SpeakerTest)[1:-1]+')'
    nameoffile = 'STCNN'+'_'+ speakerlist +'_'+str(v1)+','+str(v2)+'_'+str(SR)+'_'+str(NFRAMES)
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

def load_data(datapath , view):
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

    # first input model
    visible1 = Input(shape=(CHANNELS*COLORS,FRAME_ROWS,FRAME_COLS))

    conv11 = (Convolution2D(32, 3, border_mode='same',data_format='channels_first',
                            init='he_normal',kernel_regularizer=l2(reg)))(visible1)
    batch11 = (BatchNormalization())(conv11)
    LR11 = (LeakyReLU())(batch11)
    MP11 = LR11
    DO11 = MP11

    conv12 = (Convolution2D(64, 3, padding='same', data_format='channels_first',
                            init='he_normal',kernel_regularizer=l2(reg)))(DO11)
    batch12 = (BatchNormalization())(conv12)
    LR12 = (LeakyReLU())(batch12)
    MP12 = (AveragePooling2D(nb_pool,data_format='channels_first'))(LR12)
    DO12 = MP12

    conv13 = (Convolution2D(64, 3, padding='same', data_format='channels_first',
                            init='he_normal',kernel_regularizer=l2(reg)))(DO12)
    batch13 = (BatchNormalization())(conv13)
    LR13 = (LeakyReLU())(batch13)
    MP13 = LR13
    DO13 = MP13

    conv14 = (Convolution2D(128, 3, padding='same', data_format='channels_first',
                            init='he_normal',kernel_regularizer=l2(reg)))(DO13)
    batch14 = (BatchNormalization())(conv14)
    LR14 = (LeakyReLU())(batch14)
    MP14 = LR14
    DO14 = MP14

    conv15 = (Convolution2D(128, 3, padding='same', data_format='channels_first',
                            init='he_normal',kernel_regularizer=l2(reg)))(DO14)
    batch15 = (BatchNormalization())(conv15)
    LR15 = (LeakyReLU())(batch15)
    MP15 = LR15
    DO15 = MP15

    flat11 = (Flatten())(DO15)

    
    
    # second input model
    visible2 = Input(shape=(CHANNELS*COLORS,FRAME_ROWS,FRAME_COLS))

    conv21 = (Convolution2D(32, 3, border_mode='same',data_format='channels_first',
                            init='he_normal',kernel_regularizer=l2(reg)))(visible2)
    batch21= (BatchNormalization())(conv21)
    LR21 = (LeakyReLU())(batch21)
    MP21 = LR21
    DO21 = MP21

    conv22 = (Convolution2D(64, 3, padding='same', data_format='channels_first',
                            init='he_normal',kernel_regularizer=l2(reg)))(DO21)
    batch22 = (BatchNormalization())(conv22)
    LR22 = (LeakyReLU())(batch22)
    MP22 = (AveragePooling2D(nb_pool,data_format='channels_first'))(LR22)
    DO22 = MP22

    conv23 = (Convolution2D(64, 3, padding='same', data_format='channels_first',
                            init='he_normal',kernel_regularizer=l2(reg)))(DO22)
    batch23 = (BatchNormalization())(conv23)
    LR23 = (LeakyReLU())(batch23)
    MP23 = LR23
    DO23 = MP23

    conv24 = (Convolution2D(128, 3, padding='same', data_format='channels_first',
                            init='he_normal',kernel_regularizer=l2(reg)))(DO23)
    batch24 = (BatchNormalization())(conv24)
    LR24 = (LeakyReLU())(batch24)
    MP24 = LR24
    DO24 = MP24

    conv25 = (Convolution2D(128, 3, padding='same', data_format='channels_first',
                            init='he_normal',kernel_regularizer=l2(reg)))(DO24)
    batch25 = (BatchNormalization())(conv25)
    LR25 = (LeakyReLU())(batch25)
    MP25 = LR25
    DO25 = MP25
    flat21 = (Flatten())(DO25)

    
    # merge input models
    merge = concatenate([flat11, flat21])
    D = (Dense(128, init='he_normal'))(merge)
    batch =(BatchNormalization())(D)
    D2 = Dense(net_out, init='he_normal', use_bias=True)(batch)
    L = Activation('linear')(D2)
    model = Model(inputs=[visible1, visible2], outputs=L)
    print(model.summary())
    
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

def train_net(model1, Xtr_v1, Ytr_norm_v1, Xte_v1, Yte_norm_v1,
              Xtr_v2, Ytr_norm_v2, Xte_v2, Yte_norm_v2,
              batch_size=BATCH_SIZE, epochs=EPOCHS, finetune=False):
    if finetune:
        lr = LR/10
    else:
        lr = LR
    adam = Adam(lr=lr)
    model1.compile(loss='mse', optimizer=adam)
    if finetune:
        epochs = FINETUNE_EPOCHS
    history = model1.fit( [Xtr_v1, Xtr_v2], Ytr_norm_v1, shuffle=False,batch_size=batch_size,nb_epoch=epochs,
                         verbose=1, validation_data=([Xte_v1, Xte_v2], Yte_norm_v1))
    return model1

def predict(model, X_v1, Y_means_v1, Y_stds_v1, X_v2, Y_means_v2, Y_stds_v2, batch_size=BATCH_SIZE):
    Y_pred = model.predict([X_v1,X_v2], batch_size=batch_size, verbose=1)
    Y_pred = (Y_pred*Y_stds_v1+Y_means_v1)
    return Y_pred

def main():
		for v1 in range(1,2):
			print('****************************************')
			print('View = ',v1)
			viddata_v1, auddata_v1 = load_data(datapath,v1)
			print(viddata_v1.shape)
			(Xtr_v1,Ytr_v1), (Xte_v1, Yte_v1) = split_data(viddata_v1, auddata_v1)
			print(Xtr_v1.shape, Ytr_v1.shape)
			print(Xte_v1.shape, Yte_v1.shape)
			#sys.exit()
			Xtr_v1, Ytr_norm_v1, Xte_v1, Yte_norm_v1, Y_means_v1, Y_stds_v1 = standardize_data(Xtr_v1, Ytr_v1, Xte_v1, Yte_v1)
			for v2 in range(v1+1,3):
				print('########################################')
				print('View = ',v2)
				viddata_v2, auddata_v2 = load_data(datapath,v2)
				print(viddata_v2.shape)
				(Xtr_v2,Ytr_v2), (Xte_v2, Yte_v2) = split_data(viddata_v2, auddata_v2)
				print(Xtr_v2.shape, Ytr_v2.shape)
				print(Xte_v2.shape, Yte_v2.shape)
				Xtr_v2, Ytr_norm_v2, Xte_v2, Yte_norm_v2, Y_means_v2, Y_stds_v2 = standardize_data(Xtr_v2, Ytr_v2, Xte_v2, Yte_v2)
				model = build_model(net_out)
				model = train_net(model, Xtr_v1, Ytr_norm_v1, Xte_v1, Yte_norm_v1,
						     Xtr_v2, Ytr_norm_v2, Xte_v2, Yte_norm_v2)
				model = train_net(model, Xtr_v1, Ytr_norm_v1, Xte_v1, Yte_norm_v1,
						     Xtr_v2, Ytr_norm_v2, Xte_v2, Yte_norm_v2, finetune = True)
				Ytr_pred = predict(model, Xtr_v1, Y_means_v1, Y_stds_v1,
						      Xtr_v2, Y_means_v2, Y_stds_v2)
				Yte_pred = predict(model, Xte_v1, Y_means_v1, Y_stds_v1,
						      Xte_v2, Y_means_v2, Y_stds_v2)
				savedata(Ytr_v1, Ytr_pred, Yte_v1, Yte_pred,v1 ,v2)

if __name__ == "__main__":
    main()

