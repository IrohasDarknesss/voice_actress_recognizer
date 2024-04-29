import os
import glob
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np

class R3birthDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []
        self.label_dict = {}
        
        for index, folder in enumerate(sorted(os.listdir(root_dir))):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                self.label_dict[folder] = index
                for file in glob.glob(os.path.join(folder_path, "*.wav")):
                    self.file_paths.append(file)
                    self.labels.append(index)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.file_paths[idx])
        label = self.labels[idx]

        if self.transform:
            waveform = self.transform(waveform, sample_rate)

        return waveform, label
    
class AudioTransform:
    def __init__(self, new_sample_rate=22050):
        self.new_sample_rate = new_sample_rate
        self.resample = T.Resample(orig_freq=new_sample_rate, new_freq=self.new_sample_rate)
        self.mfcc = T.MFCC(sample_rate=self.new_sample_rate)

    def __call__(self, waveform, sample_rate):
        waveform = self.resample(waveform)
        mfcc = self.mfcc(waveform)
        return mfcc

class DataAugmentation:
    def __init__(self):
        self.time_masking = T.TimeMasking(time_mask_param=80)
        self.frequency_masking = T.FrequencyMasking(freq_mask_param=30)

    def __call__(self, mfcc):
        # Mel-frequency cepstrum
        mfcc = self.time_masking(mfcc)
        mfcc = self.frequency_masking(mfcc)
        return mfcc
    
class AudioVisualizer:
    def plot_waveform(self, waveform, sample_rate):
        plt.figure(figsize=(12, 4))
        plt.plot(np.linspace(0, len(waveform[0]) / sample_rate, num=len(waveform[0])), waveform[0])
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()
    
    def plot_spectgram(self, waveform, sample_rate):
        spectgram = T.Spectgram()(waveform)
        plt.figure(figsize=(12, 4))
        plt.imshow(spectgram.log2()[0,:,:].numpy(), cmap='viridis', aspect='auto', origin='lower')
        plt.title('Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Freqency')
        plt.colorbar(format='%+2.0f dB')
        plt.show()
                    