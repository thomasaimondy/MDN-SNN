import torch
import torchvision
from torchvision import transforms,datasets
import numpy as np
import os
import sys
import subprocess
from python_speech_features import fbank
import numpy as np
import scipy.io.wavfile as wav
from sklearn.preprocessing import normalize
import os
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import scipy.io as sio

def read_data(path, n_bands, n_frames):
    overlap = 0.5
    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.waV') and file[0] != 'O':
                filelist.append(os.path.join(root, file))
    n_samples = len(filelist)

    def keyfunc(x):
        s = x.split('/')
        return (s[-1][0], s[-2], s[-1][1]) # BH/1A_endpt.wav: sort by '1', 'BH', 'A'
    filelist.sort(key=keyfunc)

    feats = np.empty((n_samples, 1, n_bands, n_frames))
    labels = np.empty((n_samples,), dtype=np.long)
    for i, file in enumerate(filelist):
        label = file.split('/')[-1][0]  # if using windows, change / into \\
        if label == 'Z':
            labels[i] = np.long(0)
        else:
            labels[i] = np.long(label)
        rate, sig = wav.read(file)
        duration = sig.size / rate
        winlen = duration / (n_frames * (1 - overlap) + overlap)
        winstep = winlen * (1 - overlap)
        feat, energy = fbank(sig, rate, winlen, winstep, nfilt=n_bands, nfft=4096, winfunc=np.hamming)
        # feat = np.log(feat)
        final_feat = feat[:n_frames]
        final_feat = normalize(final_feat, norm='l1', axis=0)
        a = np.expand_dims(np.array(final_feat), axis=0)
        feats[i] = a
    # feats[i] = feat[:n_frames].flatten() # feat may have 41 or 42 frames
    # feats[i] = feat.flatten() # feat may have 41 or 42 frames

    # feats = normalize(feats, norm='l2', axis=1)
    # normalization
    # feats = preprocessing.scale(feats)

    np.random.seed(42)
    p = np.random.permutation(n_samples)
    feats, labels = feats[p], labels[p]

    n_train_samples = int(n_samples * 0.7)

    train_set = (feats[:n_train_samples], labels[:n_train_samples])
    test_set = (feats[n_train_samples:], labels[n_train_samples:])

    return train_set, train_set, test_set

class Tidigits(Dataset):
    def __init__(self,train_or_test,input_channel,n_bands,n_frames,transform=None, target_transform = None):
        super(Tidigits, self).__init__()
        self.n_bands = n_bands
        self.n_frames = n_frames
        dataname = 'tidigits/packed_tidigits_nbands_'+str(n_bands)+'_nframes_' + str(n_frames)+'.pkl'
        if os.path.exists(dataname):
            with open(dataname,'rb') as fr:
                [train_set, val_set, test_set] = pickle.load(fr)
        else:
            print('Tidigits Dataset Has not been Processed, now do it.')
            train_set, val_set, test_set = read_data(path='tidigits/isolated_digits_tidigits', n_bands=n_bands, n_frames=n_frames)
            with open(dataname,'wb') as fw:
                pickle.dump([train_set, val_set, test_set], fw)
        if train_or_test == 'train':
            self.x_values = train_set[0]
            self.y_values = train_set[1]

        elif train_or_test == 'test':
            self.x_values = test_set[0]
            self.y_values = test_set[1]
        elif train_or_test == 'valid':
            self.x_values = val_set[0]
            self.y_values = val_set[1]
        self.transform =transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample = self.x_values[index]
        label = self.y_values[index]
        return sample, label

    def __len__(self):
        return len(self.x_values)