import os
import wavio

import torch
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline

import librosa
import librosa.display

import pdb
#from loader import get_spectrogram_feature  as naver

N_FFT = 512
N_MELS = 40
SAMPLE_RATE = 16000

def get_feature_from_librosa(filepath, feature_size, sample_rate):
    hop_length = 128

    sig, sampling = librosa.core.load(filepath, sample_rate)

    #assert sample_rate == 16000

    mfcc_feat = librosa.feature.mfcc(y=sig, sr=sample_rate, hop_length=hop_length, n_mfcc=feature_size, n_fcc=512)

    return mfcc_feat

def get_mel_feature_from_librosa(filepath, type_):
    hop_length = int(0.01*SAMPLE_RATE)

    sig, sampling = librosa.core.load(filepath, SAMPLE_RATE)
    #assert sample_rate == SAMPLE_RATE

    _mel = librosa.feature.melspectrogram(sig, n_mels=N_MELS, n_fft=N_FFT, hop_length=hop_length)
    
    if type_ == 'mel':
        return _mel
    
    elif type_ == 'log_mel':
        log_mel = librosa.amplitude_to_db(_mel, ref = np.max)
        return log_mel

def get_spectrogram_feature_modify(filepath):
    (rate, width, sig) = wavio.readwav(filepath)
    sig = sig.ravel()
    
    stft = torch.stft(torch.FloatTensor(sig),
                    N_FFT,
                    hop_length=int(0.01*SAMPLE_RATE),
                    win_length=int(0.030*SAMPLE_RATE),
                    window=torch.hamming_window(int(0.030*SAMPLE_RATE)),
                    center=False,
                    normalized=False,
                    onesided=True)

    stft = (stft[:,:,0].pow(2) + stft[:,:,1].pow(2)).pow(0.5);
    amag = stft.numpy()
    feat = torch.FloatTensor(amag)
    feat = torch.FloatTensor(feat).transpose(0,1)

    return feat, sig #feat : [time, 257], sig: []


def draw_image(feat):
    plt.figure(figsize=(12,4))
    plt.subplot(211)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.imshow(feat, aspect='auto', origin='lower')
     

if __name__ == "__main__":
    sample_data = './sample_dataset/train/train_data'
    data_list = os.path.join(sample_data, 'data_list.csv')
    
    wav_paths = list()
    script_paths = list()

    with open(data_list, 'r') as f:
        for line in f:
            wav_path, script_path = line.strip().split(',')
            wav_paths.append(os.path.join(sample_data, wav_path))
            script_paths.append(os.path.join(sample_data, script_path))

    naver_test = get_spectrogram_feature_modify(wav_paths[0])
    mel_test = get_mel_feature_from_librosa(wav_paths[0])
    #get_spectrogram_feature, get_mel_feature_from_librosa sig outcome 동일
    #pdb.set_trace()
