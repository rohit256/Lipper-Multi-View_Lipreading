import os
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import sys
import audio_tools as aud
import audio_read as ar
import librosa
import wave
import contextlib
import moduletest
from moviepy.editor import VideoFileClip


SR = moduletest.SamplingRate
FRAME_ROWS = 128
FRAME_COLS = 128
NFRAMES = moduletest.NumberOfFrames  # size of consecutive frames for feeding NN
MARGIN = int(NFRAMES/2)
COLORS = 1 # grayscale
CHANNELS = NFRAMES
OVERLAP = 1.0/2
LPC_ORDER = moduletest.LPCOrder
NET_OUT = int(1/OVERLAP)*(LPC_ORDER+1)
SEQ_LEN = 30
NUM_VIDS = len(moduletest.Audios)
SAMPLE_LEN = SEQ_LEN-2*MARGIN  #Same as SEQ_LEN - NFRAMES + 1
MAX_FRAMES = NUM_VIDS*COLORS*SAMPLE_LEN
audpath = '../../lipsync/dataset/cropped_audio_dat/'
#audpath = '../dataset/cropped_audio_dat/'
def process_video(vf, viddata, vidctr):
    temp_frames = np.zeros((SEQ_LEN,FRAME_ROWS,FRAME_COLS),dtype="uint8")
    cap = cv2.VideoCapture(vf)
    t = (VideoFileClip(vf).duration*1000)/SEQ_LEN
    cap = cv2.VideoCapture(vf)
    for i in np.arange(SEQ_LEN):
        cap.set(cv2.CAP_PROP_POS_MSEC,t*i)
        ret,frame = cap.read()
        if ret==0:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face = frame
        face = cv2.resize(face,(FRAME_COLS,FRAME_ROWS))
        face = np.expand_dims(face,axis=2)
        face = face.transpose((2,0,1))
        temp_frames[i,:,:] = face
    for i in np.arange(MARGIN,SAMPLE_LEN+MARGIN):
        viddata[vidctr,:,:,:] = temp_frames[(i-MARGIN):(i+MARGIN+1),:,:]
        vidctr = vidctr+1
    print(viddata.shape)
    return vidctr

def process_audio(af, auddata, audctr):
    print (af)
    (y,sr) = librosa.load(af,sr=SR)
    with contextlib.closing(wave.open(af,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    win_length = int((SR*duration)/SEQ_LEN)
    hop_length = int(win_length*OVERLAP)
    [a,g,e] = aud.lpc_analysis(y,LPC_ORDER,window_step=hop_length,window_size=win_length)
    lsf = aud.lpc_to_lsf(a)
    lsf = lsf[(MARGIN)*int(1/OVERLAP):(SAMPLE_LEN+MARGIN)*int(1/OVERLAP),:]
    lsf_concat = np.concatenate((lsf[::2,:],lsf[1::2,:]),axis=1) # MAGIC NUMBERS for half overlap
    g = g[(MARGIN)*int(1/OVERLAP):(SAMPLE_LEN+MARGIN)*int(1/OVERLAP),:]     
    g_concat = np.concatenate((g[::2,:],g[1::2,:]),axis=1) # MAGIC NUMBERS for half overlap
    feat = np.concatenate((lsf_concat,g_concat),axis=1)
    auddata[audctr:audctr+SAMPLE_LEN,:] = feat
    audctr = audctr+SAMPLE_LEN
    return audctr

def show_progress(progress, step):
    progress += step
    sys.stdout.write("Processing progress: %d%% \r"%(int(progress)))
    sys.stdout.flush()
    return progress

def createNumpyArrays(speaker,view):

	apath='../../lipsync/dataset/numpy_datasets/auddata_'+str(SR)+'_'+str(NFRAMES)+'_'+str(speaker)+'.npy'
	vpath='../../lipsync/dataset/numpy_datasets/viddata_'+str(SR)+'_'+str(NFRAMES)+'_'+str(speaker)+'_'+str(view)+'.npy'
	videopath='../../lipsync/dataset/cropped_mouth_mp4_phrase/'+str(speaker)+'/'+str(view)+'/s'+str(speaker)+'_v'+str(view)+'_u'
	audiopath='../../lipsync/dataset/cropped_audio_dat/s'+str(speaker)+'_u'
	#apath='../dataset/numpy_datasets/auddata_'+str(SR)+'_'+str(NFRAMES)+'_'+str(speaker)+'.npy'
	#vpath='../dataset/numpy_datasets/viddata_'+str(SR)+'_'+str(NFRAMES)+'_'+str(speaker)+'_'+str(view)+'.npy'
	#videopath='../dataset/cropped_mouth_mp4_phrase/'+str(speaker)+'/'+str(view)+'/s'+str(speaker)+'_v'+str(view)+'_u'
	#audiopath='../dataset/cropped_audio_dat/s'+str(speaker)+'_u'

	if(os.path.exists(apath) and os.path.exists(vpath)):
		return;
	viddata = np.zeros((MAX_FRAMES,CHANNELS,FRAME_ROWS,FRAME_COLS),dtype="uint8")
	auddata = np.zeros((MAX_FRAMES,NET_OUT),dtype="float32")
	print(MAX_FRAMES) 
	vidctr=0
	audctr=0
	sorted_vids=[]
	sorted_vids += moduletest.Audios
	for j in sorted_vids:
	    #print 'hello'
	    #print str(j)
	    vidctr = process_video(videopath+str(j)+'.mp4', viddata, vidctr)
	    audctr = process_audio(audiopath+str(j)+'.wav', auddata, audctr)
	print ('Done processing. Saving data to disk...')
	viddata = viddata[:vidctr,:,:,:]
	auddata = auddata[:audctr,:]
	print(viddata.shape)
	print(auddata.shape)
	np.save(vpath, viddata)
	np.save(apath, auddata)
	print ('Finally Done !!!!!!!!!!!')
            
def main():

    for speaker in moduletest.SpeakerTrain:

        for view in range(1,6):

            createNumpyArrays(speaker,view)
        
    for speaker in moduletest.SpeakerTest:

        for view in range(1,6):

            createNumpyArrays(speaker,view)

    return

if __name__ == "__main__":
    main()
salik@falcon
