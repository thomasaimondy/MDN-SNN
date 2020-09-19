import numpy as np
import glob
import os
import re
import random
import editdistance
import sys
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import matplotlib.pyplot as plt
from sphfile import SPHFile

class Timit(Dataset):
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", ',', ';', '"', ':', '!', '.']

    def __init__(self, phase, transform=None):
        super(Timit, self).__init__()
        self.audio_path = 'data/Timit'
        self.audio_pad = 200
        self.txt_pad = 90
        self.phase = phase

        self.audios = glob.glob(os.path.join(self.audio_path, self.phase, '*', '*', '*.wav'))
        self.data = []
        self.mfcc_l = []
        self.transform = transform
        pad = 0

        male = None
        female = None
        dataname = self.audio_path + '/' + self.phase + 'check_gender.pkl'
        if os.path.exists(dataname):
            with open(dataname, 'rb') as fr:
                self.data = pickle.load(fr)
        else:
            for audio in self.audios:
                fs, sound = wav.read(audio)
                mfcc_features = mfcc(sound, samplerate = 16000)
                mfcc_features = self._padding(mfcc_features, 800)

                items = audio.strip('').split('.')

                txt = items[0] + '.TXT'
                if self.phase == 'TRAIN':
                    if items[0][21] == 'M':
                        gender = 1
                        if male is None:
                            male = items[0][22:26]
                            print(male)
                            print('First')
                    elif items[0][21] == 'F':
                        gender = 0
                        if female is None:
                            female = items[0][22:26]
                            print(female)
                            print('First')
                    dialect = int(items[0][19]) - 1
                elif self.phase == 'TEST':
                    if items[0][20] == 'M':
                        gender = 1
                        if male is None:
                            male = items[0][21:25]
                    elif items[0][20] == 'F':
                        gender = 0
                        if female is None:
                            female = items[0][21:25]
                    dialect = int(items[0][18]) - 1

                self.data.append((mfcc_features, gender, dialect))

                with open(dataname, 'wb') as fw:
                    pickle.dump(self.data, fw)

    def __getitem__(self, idx):
        (mfcc_features, gender, dialect) = self.data[idx]
        mfcc_features = torch.FloatTensor(mfcc_features)
        label = gender
        mfcc_features = mfcc_features.view(20, -1)

        return torch.FloatTensor(mfcc_features), label

    def __len__(self):
        return len(self.data)

    def _load_audio(self, f):
        fs, sound = wav.read(f)
        mfcc_features = mfcc(sound, samplerate = 16000)

        return mfcc_features

    def _load_text(self, text):
        print(text)
        with open(text, 'r') as f:
            lines = [line.strip().upper() for line in f.readlines()]
            lines_list = list(lines[0])
            print(''.join(lines_list))
            lines_list = lines_list[8:-2]

        return Timit.txt2arr(' '.join(lines_list).upper(), 1)

    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis = 0)