import os
import math
import h5py
import random
import numpy as np
import scipy.io as scio
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers import Input,BatchNormalization,Lambda,Concatenate,Multiply
from tensorflow.python.keras.models import Model,Sequential,load_model
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import LearningRateScheduler,ModelCheckpoint,EarlyStopping,TensorBoard
from tensorflow.python.keras.initializers import he_uniform
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA 
from scipy.stats import entropy,norm
import tensorflow.compat.v1 as tf
import pandas as pd
tf.compat.v1.disable_eager_execution()
from numpy import sqrt, newaxis
from numpy.fft import irfft, rfftfreq
from numpy.random import normal
from numpy import sum as npsum
from scipy import signal
from scipy.spatial import distance
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True 
import librosa
import time
start = time.time()

def enframe(wavData, frameSize, overlap):
    wlen = len(wavData)
    step = frameSize - overlap
    frameNum:int = math.ceil(wlen / step)
    frameData = np.zeros((frameSize, frameNum-4))
    hamwin = np.hamming(frameSize)   
    for i in range(frameNum-4):
        singleFrame = wavData[np.arange(i * step, i * step+frameSize)]      
        frameData[:len(singleFrame),i] = singleFrame
    return frameData


def load_ssdata(matname,keyname):
    matdata = scio.loadmat(matname)
    data = matdata[keyname]
#     data = data[shuffle_index,:]
    length = len(data)
    ss_input_data = enframe(data[0,:].T,6400,5120)
    for i in  range(1,length):
        wave_data = data[i,:].T
        framedata = enframe(wave_data,6400,5120)
        ss_input_data = np.hstack((ss_input_data,framedata))
    return ss_input_data.T  #shape : (200*20*num,800)

def load_eegdata(matname,keyname,channels):
    matdata = scio.loadmat(matname)
#     matdata.keys()
    data = matdata[keyname]
#     data = data[:,:,shuffle_index]
    length = data.shape[2]
    for i in range(0,length):
        temp_epoch = data[:,:,i].T
        temp_frame = enframe(temp_epoch[:,0].T,400,320)
#         print(temp_frame.shape)
        for k in  range(1,channels): 
            temp_channel = temp_epoch[:,k].T
            temp_frame = np.vstack((temp_frame,enframe(temp_channel,400,320)))
        if i == 0:
            frame_all = temp_frame
        else:
            frame_all = np.hstack((frame_all,temp_frame))

    return frame_all

#Dynamically adjust the learning rate
def step_decay1(epoch):                 #固定指数降低
    initial_lrate =1e-4
    drop_rate = 0.9
    epochs_drop =1000.0
    # LearningRate = InitialLearningRate * DropRate^floor(Epoch / EpochDrop)
    lr = initial_lrate * math.pow(drop_rate, math.floor((1+epoch)/epochs_drop))
    return lr

def step_decay2(epoch):                 #固定指数降低
    initial_lrate =1e-4
    drop_rate = 0.9
    epochs_drop =2000.0
    # LearningRate = InitialLearningRate * DropRate^floor(Epoch / EpochDrop)
    lr = initial_lrate * math.pow(drop_rate, math.floor((1+epoch)/epochs_drop))
    return lr

def build_single_model(name_add,hlayer_num,rate,nb_input):
    
    model = Sequential()
    model.add(Input(shape=(nb_input,),name='input'+name_add))
    # model.add(BatchNormalization(axis=1))
    nb_hidden = int(nb_input*(rate/100))
    model.add(Dense(units=int(nb_hidden*pow(rate/100,0)),kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name=str(0)+'_en'+name_add))
    #Grid Search for layer number (Symmetrical structure)
    for i in range(1,hlayer_num):                        
        model.add(Dense(units=int(nb_hidden*pow(rate/100,i)),kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name=str(i)+'_en'+name_add))

    model.add(Dense(units=int(nb_hidden*pow(rate/100,hlayer_num)),kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name='middle'+name_add))

    for j in range(hlayer_num-1,0,-1):
        model.add(Dense(units=int(nb_hidden*pow(rate/100,j)),kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name=str(j)+'_de'+name_add))
    
    model.add(Dense(units=int(nb_hidden*pow(rate/100,0)),kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name='0_de'+name_add))

    model.add(Dense(units=nb_input,activation='tanh',kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',name='output'+name_add))
    # Compile model
    adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
    model.compile(loss='mse', optimizer=adm)
    model.summary()
    return model

def build_ETS_ld_fusion_model(ss_layer_num,eeg_layer_num,sh,sb,eh,eb):

    nb_ss_input = 6400
    nb_eeg_input = 18000
    
    ss_input_layer = Input(shape=(nb_ss_input,),name='ss_input')
    ss_hidden = ss_model.layers[0](ss_input_layer)
    eeg_input_layer = Input(shape=(nb_eeg_input,),name='eeg_input')
    eeg_hidden = ee_model.layers[0](eeg_input_layer)
   

    ss_hidden = ss_model.layers[1](ss_hidden)
    eeg_hidden = ee_model.layers[1](eeg_hidden)
    
    #fusion second time
    con_layer2 =  Concatenate()([ss_hidden, eeg_hidden])
    ss_con_shape = ss_model.layers[1].output_shape[1]
    ee_con_shape = ee_model.layers[1].output_shape[1]
    con_shape = int((ss_con_shape+ee_con_shape)*(sb+eb)/100)
    con_fusion3 = Dense(units=con_shape,kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='ss_con_2')(con_layer2)
    con_fusion4 = Dense(units=int(con_shape*sb/100),kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='ee_con_2')(con_fusion3)
    con_fusion4 = Dense(units=int(ss_con_shape+ee_con_shape),kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='con_sym2')(con_fusion4)
    ss_sym_h = Dense(units=ss_con_shape,kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='ss_sym_2')(con_fusion4)
    ee_sym_h = Dense(units=ee_con_shape,kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='ee_sym_2')(con_fusion4)
    ss_hidden = ss_model.layers[2](ss_sym_h)
    eeg_hidden = ee_model.layers[2](ee_sym_h)
 
   
    #fusion third time
    con_layer3 =  Concatenate()([ss_hidden, eeg_hidden])
    ss_con_shape = ss_model.layers[ss_layer_num-1].output_shape[1]
    ee_con_shape = ee_model.layers[eeg_layer_num-1].output_shape[1]

    con_shape = int((ss_con_shape+ee_con_shape)*(sb+eb)/100)
    con_fusion5 = Dense(units=con_shape,kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='ss_con_3')(con_layer3)
    con_fusion6 = Dense(units=int(con_shape*sb/100),kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='ee_con_3')(con_fusion5)
    re_fusion = Dense(units=con_shape,kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='refucion')(con_fusion6)
    re_con = Dense(units=int(ss_con_shape+ee_con_shape),kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='recon')(re_fusion)
   
    ss_hidden = Dense(units=ss_con_shape,kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='ss_sym_3')(re_con)
    eeg_hidden = Dense(units=ee_con_shape,kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='ee_sym_3')(re_con)
   
   
    for i in range(ss_layer_num,len(ss_model.layers)-1):
        ss_hidden = ss_model.layers[i](ss_hidden)
    for i in range(eeg_layer_num,len(ee_model.layers)-1):
        eeg_hidden = ee_model.layers[i](eeg_hidden)
    
    #ETS Model Output Layers
    ss_output_layer = Dense(units=nb_ss_input,name='ss_output',activation='tanh',bias_initializer='TruncatedNormal',kernel_initializer='glorot_uniform')(ss_hidden)
    eeg_output_layer = Dense(units=nb_eeg_input,name='eeg_output',activation='tanh',bias_initializer='TruncatedNormal',kernel_initializer='glorot_uniform')(eeg_hidden)
    
    model=Model(inputs=[ss_input_layer,eeg_input_layer],outputs=[ss_output_layer,eeg_output_layer])
    model.summary()
    return model         

def build_eeg_gandy(rate,nb_input):
    green_in = Input(shape=(nb_input,),name='green_input')
    green_hid = Dense(units=int(nb_input*pow(rate/100,1)),kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name='0_green_en')(green_in)
    yellow_in = Input(shape=(nb_input,),name='yellow_input')
    yellow_hid = Dense(units=int(nb_input*pow(rate/100,1)),kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name='0_yellow_en')(yellow_in)
    green_hid = Dense(units=int(nb_input*pow(rate/100,2)),kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name='1_green_en')(green_hid)
    yellow_hid = Dense(units=int(nb_input*pow(rate/100,2)),kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name='1_yellow_en')(yellow_hid)
    green_hid = Dense(units=int(nb_input*pow(rate/100,3)),kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name='2_green_en')(green_hid)
    yellow_hid = Dense(units=int(nb_input*pow(rate/100,3)),kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name='2_yellow_en')(yellow_hid)

    #con
    con_layer =  Concatenate()([green_hid, yellow_hid])

    fusion_rate = 0.75
    con_shape = int((nb_input*pow(rate/100,3))*2*fusion_rate)
    con_fusionin = Dense(units=con_shape,kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='fusion_in')(con_layer)
    con_fusionmid = Dense(units=int(con_shape*fusion_rate),kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='fusion_middle')(con_fusionin)
    re_fusion = Dense(units=con_shape,kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='fusion_out')(con_fusionmid)
    re_con = Dense(units=int(nb_input*pow(rate/100,3))*2,kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='recon')(re_fusion)
   
    green_hidden = Dense(units= int(nb_input*pow(rate/100,3)),kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='ss_sym_3')(re_con)
    yellow_hidden = Dense(units= int(nb_input*pow(rate/100,3)),kernel_initializer='glorot_uniform',activation='tanh',bias_initializer='TruncatedNormal',name='ee_sym_3')(re_con)
   

    ##decoder
    con_shape1 = int(nb_input*pow(rate/100,1))
    con_shape2 = int(nb_input*pow(rate/100,2))
    green_hid =  Dense(units=con_shape2,kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name='1_green_dec')(green_hidden)
    yellow_hid = Dense(units=con_shape2,kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name='1_yellow_dec')(yellow_hidden)
    green_hid =  Dense(units=con_shape1,kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name='0_green_dec')(green_hid)
    yellow_hid = Dense(units=con_shape1,kernel_initializer='glorot_uniform',bias_initializer='TruncatedNormal',activation='tanh',name='0_yellow_dec')(yellow_hid)
    
    
    green_out = Dense(units=nb_input,name='green_output',activation='tanh',bias_initializer='TruncatedNormal',kernel_initializer='glorot_uniform')(green_hid)
    yellow_out = Dense(units=nb_input,name='yellow_output',activation='tanh',bias_initializer='TruncatedNormal',kernel_initializer='glorot_uniform')(yellow_hid)
    
    model=Model(inputs=[green_in,yellow_in],outputs=[green_out,yellow_out])
#     model.summary()
    return model         

def compute_error(green_pre,yellow_pre,lenth):
    # lenth = int(time/frame_shift-4)
    error_ee = np.zeros((lenth,lenth)) 
   
    for i in range(lenth):
        for j in range(lenth):
            # error_ee[i,j] = distance.euclidean(green_pre[i,:],yellow_pre[j,:])
            temp = np.corrcoef(np.array([green_pre[i,:],yellow_pre[j,:]]))
            error_ee[i,j] = -np.abs(temp[1,0])
    
    return error_ee

def find_match_dtw(error_matrix,data_green,data_yellow,lenth):
    
    data_match_green,data_match_yellow= [],[]
   
    iframe_index = []
    cost = np.zeros((lenth,lenth))

    # Initialize the first row and column
    cost[0, 0] = error_matrix[0,0]
    for i in range(1, lenth):
        cost[i, 0] = cost[i-1, 0] + error_matrix[i,0]

    for j in range(1, lenth):
        cost[0, j] = cost[0, j-1] + error_matrix[0,j]

    # Populate rest of cost matrix within window
    for i in range(1,lenth):
        for j in range(1,lenth):
            choices = [cost[i-1, j-1], cost[i, j-1], cost[i-1, j]]
            cost[i, j] = min(choices) + error_matrix[i,j]
    
    iframe_index.append([lenth-1,lenth-1])
    data_match_green.append(data_green[lenth-1,:])
    data_match_yellow.append(data_yellow[lenth-1,:])
    
#     data_match[lenth-1,:] = data_ori[lenth-1,:]
    i,j = lenth-1,lenth-1
    while(i>0 and j > 0):
        
        index = np.argmin([cost[i, j-1], cost[i-1, j-1], cost[i-1, j]])
        if index==1 :
            data_match_green.append(data_green[i-1,:])
            data_match_yellow.append(data_yellow[j-1,:])
            iframe_index.append([i-1,j-1])
            i-=1
            j-=1
        elif index==2:
            data_match_green.append(data_green[i-1,:])
            data_match_yellow.append(data_yellow[j,:])
            iframe_index.append([i-1,j])
           
            i-=1   
        else:
            data_match_green.append(data_green[i,:])
            data_match_yellow.append(data_yellow[j-1,:])
            iframe_index.append([i,j-1])
            j-=1   
    while(i==0 and j!=0):
        data_match_green.append(data_green[i,:])
        data_match_yellow.append(data_yellow[j-1,:])
        iframe_index.append([i,j-1])
        j-=1   
    
    while(j==0 and i!=0):
        data_match_green.append(data_green[i-1,:])
        data_match_yellow.append(data_yellow[j,:])
        iframe_index.append([i-1,j])
        i-=1   
    # print( iframe_index)
    return np.array(data_match_green),np.array(data_match_yellow),iframe_index

def wgn(x, snr):
    Ps = np.sum(abs(x)**2)/len(x)
    Pn = Ps/(10**((snr/10)))
    noise = np.random.randn(len(x)) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise

def signal_xHz(A, fi, time_s, sample):
    return A * np.sin(np.linspace(0, fi * time_s * 2 * np.pi , sample* time_s))

def generate_sin():
    n,m = 16000,2016
    test_sin = np.zeros((n,m))
    for i in range(m): 
        for j in range(10):
            test_sin[:,i]+=signal_xHz(np.random.random(),np.random.randint(1,50), 16, 1000)
        test_sin[:,i] = test_sin[:,i]*1.0/(max(abs(test_sin[:,i])))
    return test_sin

def slimindex(ifindex,data1,data2):
    slimindex = []
    iindex = np.array(ifindex)
#     print(ifindex)
    i = 0
    slimindex.append(ifindex[i])
    while (i < len(ifindex)-2):
        k = 0

        while((i+k+1)<len(ifindex) and iindex[i+k,0]==iindex[i+k+1,0]):
            k=k+1
        
        if(i+k+1)<len(ifindex):
            i = i+k+1
        else:
            break

        slimindex.append(ifindex[i])

    data_match_green,data_match_yellow= [],[]
    newm = np.array(slimindex)
    if(newm[len(slimindex)-1,0]!=0):
        slimindex.append([0,0])
        newm= np.array(slimindex)
    
    for i in range(len(slimindex)):
        data_match_green.append(data1[newm[i,0],:])
        data_match_yellow.append(data2[newm[i,1],:])
#     print(len(slimindex),slimindex)
    return slimindex,np.array(data_match_green),np.array(data_match_yellow)


def Spectral_subtraction(data,flag_lowpass):
    #stft raw wav
    s= librosa.stft(data)    # Short-time Fourier transform
    ss= np.abs(s)         # get magnitude
    angle= np.angle(s)    # get phase
    b=np.exp(1.0j* angle) # use this phase information when Inverse Transform
    
    # Noise magnitude calculations - assuming that the first 5 frames is noise/silence
    len_= 1280
    total_frames = int(len(data)/len_)
    noise_data = np.hstack((data[len_*3:len_*6],data[len_*(total_frames-6):len_*(total_frames-3)]))
#     noise_data = noise_data.repeat(len(data)/(len_*10))
    ns= librosa.stft(noise_data) 
    nss= np.abs(ns)
    noise_mu= np.mean(nss, axis=1) # get mean
   # subtract noise spectral mean from input spectral, and istft (Inverse Short-Time Fourier Transform)
    sa= ss - noise_mu.reshape((noise_mu.shape[0],1))  # reshape for broadcast to subtract
    sa0= sa * b  # apply phase information
    y= librosa.istft(sa0) # back to time domain signal

    if(flag_lowpass==1):
        b, a =butter_lowpass(3000, 8000, order=7)
        y = signal.filtfilt(b, a, y)

    return y


def recons_ss_noise(num,reconswave,cir_k,flag_lowpass,fadd):
   
    denoise_ss = [] 
    shape_lenth = np.zeros(32000)
    for i in range(num):
        frame_shift = 1280
        temp_ss = reconswave[i]
#         temp_ss[frame_shift*2:frame_shift*24] =  Spectral_subtraction(temp_ss[frame_shift*2:frame_shift*24],flag_lowpass)
        recons_wave= Spectral_subtraction(reconswave[i],flag_lowpass)
#         cir_k =3
        for j in range(cir_k):
#               temp_ss[frame_shift*2:frame_shift*24] =  Spectral_subtraction(temp_ss[frame_shift*2:frame_shift*24],flag_lowpass)
              recons_wave= Spectral_subtraction(recons_wave,flag_lowpass)
        shape_lenth[0:len(recons_wave)] = recons_wave
#         w_data = recons_wave/(1.2*(max(abs(recons_wave))))
        w_data = shape_lenth/(1.2*(max(abs(shape_lenth))))
        denoise_ss.append(w_data)
#         recons_wave = ss_noise(recons_wave)
    return denoise_ss


def load_ssdata_specsub(matname,keyname):
    matdata = scio.loadmat(matname)
    data = matdata[keyname]
#     data = data[shuffle_index,:]
    length = len(data)
    data = np.array(recons_ss_noise(length,data,5,0,''))
    ss_input_data = enframe(data[0,:].T,6400,5120)
    for i in  range(1,length):
        wave_data = data[i,:].T
        framedata = enframe(wave_data,6400,5120)
        ss_input_data = np.hstack((ss_input_data,framedata))
    return ss_input_data.T  #shape : (200*20*num,800)


