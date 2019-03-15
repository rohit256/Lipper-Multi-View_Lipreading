import numpy as np
import os
from os import listdir
from os.path import isfile, join
import sys
import moviepy.editor as mpy
import argparse
import wave
import contextlib
import audio_tools as aud
from scipy.io.wavfile import write
import keras.backend as BK
import numpy
from scipy.stats.stats import pearsonr   
from scipy.stats import linregress
import moduletest

respath = '../results/'
SR = moduletest.SamplingRate
NFRAMES = moduletest.NumberOfFrames
MARGIN = int(NFRAMES/2)
OVERLAP = 1.0/2
LPC_ORDER = moduletest.LPCOrder
NET_OUT = int(1/OVERLAP)*(LPC_ORDER+1)
SEQ_LEN = 30
NUM_VIDS = len(moduletest.Audios)
SAMPLE_LEN = SEQ_LEN-2*MARGIN
STEP = int(SAMPLE_LEN*1/OVERLAP)

def get_lsf(Y_pred):
    lsf_tepr = Y_pred[:,:-2]
    lsf_tepr2 = np.zeros((lsf_tepr.shape[0]*2,int(lsf_tepr.shape[1]/2)))
    lsf_tepr2[::2,:] = lsf_tepr[:,:LPC_ORDER]
    lsf_tepr2[1::2,:] = lsf_tepr[:,LPC_ORDER:]
    g_tepr = Y_pred[:,-2:]
    g_tepr2 = np.zeros((g_tepr.shape[0]*2,1))
    g_tepr2[::2,:] = g_tepr[:,:1]
    g_tepr2[1::2,:] = g_tepr[:,1:]
    g_tepr2[g_tepr2<0] = 0.0
    return lsf_tepr2, g_tepr2

def order_vids(sorted_vids):
    sorted_vids += moduletest.Audios
    
def synthesize(af,lpc,g):
    with contextlib.closing(wave.open(af,'r')) as f:
             frames = f.getnframes()
             rate = f.getframerate()
             duration = frames / float(rate)
    #print duration
    win_length = int((SR*duration)/SEQ_LEN)
    hop_length = int(win_length*OVERLAP)
    x = aud.lpc_synthesis(lpc,g,None,window_step=hop_length)
    return x

def construct_name(speaker, view):
    predfile = 'Yte'   ######## Change to either Yte or Yte_pred #########
    speakerlist = '('+str(moduletest.SpeakerTrain)[1:-1] + '||||' + str(moduletest.SpeakerTest)[1:-1]+')'
    nameoffile = predfile + '_STCNN'+'_'+ speakerlist +'_'+str(view).replace(" ","")[1:-1]+'_'+str(SR)+'_'+str(NFRAMES)
    nameoffile+= '.npy'
    return nameoffile

def construct_destinationFolder(speaker,view):
	
	sample_path = join(respath,'samples/speaker_independent/'+str(moduletest.Views)+'_View/')
	respath_view = join(sample_path,'View='+str(view)[1:-1]+'/')
	if not os.path.exists(respath_view):
		os.makedirs(respath_view)
	respath_speaker = join(respath_view,'Speaker_'+str(speaker)+'/')
	if not os.path.exists(respath_speaker):
		os.makedirs(respath_speaker)
	return respath_speaker

def generate_samples(speaker,view, ctr):
    predname = construct_name(speaker,view)
    numpydata_path = join(respath+'speaker_independent/'+str(moduletest.Views)+'_View/'+
	             'View='+str(view).replace(" ","")[1:-1]+'/', predname)
    Y_pred = np.load(numpydata_path)
    print(Y_pred.shape)
    #sys.exit()
    sorted_vids = []
    order_vids(sorted_vids)
    lsf,g = get_lsf(Y_pred)
    destination_path = construct_destinationFolder(speaker,view)
    shift = ctr*NUM_VIDS
    for i in range(len(moduletest.Audios)):
        lpc_i = aud.lsf_to_lpc(lsf[(i+shift)*STEP:(i+1+shift)*STEP,:])
        g_i = g[(i+shift)*STEP:(i+1+shift)*STEP,:]
        AUDPATH = '../../lipsync/dataset/cropped_audio_dat'+'/s'+str(speaker)+'_u'
        af = AUDPATH + str(sorted_vids[i])+'.wav'
        print(af)
        x = synthesize(af,lpc_i,g_i)
        filename = str(SR)+'_'+str(NFRAMES)+'_'+str(sorted_vids[i])+'.wav'
        write(destination_path+filename,SR,x)

def main(): 
        viewcomb = moduletest.Views
        li = []
        if viewcomb == 1 :
                li = moduletest.VIEWS_1
        if viewcomb == 2:
                li = moduletest.VIEWS_2
        if viewcomb == 3:
                li == moduletest.VIEWS_3
        if viewcomb == 4:
                li == moduletest.VIEWS_4
        ctr = 0
        for speaker in moduletest.SpeakerTest:
            for view in li:
                generate_samples(speaker,view , ctr)
                ctr = ctr+1
            
if __name__ == "__main__":
    main()


