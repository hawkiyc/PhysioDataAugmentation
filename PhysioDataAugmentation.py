# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:09:04 2023

@author: Revlis_user
"""
#%%
'Import Libraries'

import numpy as np
from scipy import signal

#%%
'Data Augmentation'

class AddBaselineWander(object):
    def __init__(self, sampling_rate=50, cutoff_freq=1.1, std=.22, p = .5):
        self.sampling_rate = sampling_rate
        self.cutoff_freq = cutoff_freq
        self.std = std
        self.p = p

    def __call__(self, data):
        
        N, _, L = data.shape
        # Generate low-frequency noise
        low_freq_noise = np.random.normal(0, self.std, (N, L))
        # Apply a low-pass filter to the noise
        b, a = signal.butter(5, self.cutoff_freq, 'low', 
                             fs=self.sampling_rate)
        for i in range(N):
            if np.random.random() <= self.p:
                low_freq_noise[i,:] = signal.filtfilt(b, a, 
                                                      low_freq_noise[i,:])
            else:
                low_freq_noise[i,:] = 0
        # Add the noise to the original signal
        data_with_wander = data + low_freq_noise.reshape(N, 1, L)
        return data_with_wander
        

    def __repr__(self):
        return self.__class__.__name__ + '''(sampling_rate={0}, 
            cutoff_freq={1}, std={2}, p={3})'''.format(
                self.sampling_rate, self.cutoff_freq, self.std, self.p)

class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        N = data.shape[0]
        _ = data.clone()
        for i in range(N):
            if np.random.random() <= self.p:
                _[i,:] = -_[i,:]
        return _

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)

class AddWhiteNoise(object):
    def __init__(self, std=.2, p=0.5):
        self.p = p
        self.std = std

    def __call__(self, data):
        N, _, L = data.shape
        white_noise = np.random.normal(0, self.std, (N, L))
        _ = data.clone()
        for i in range(N):
            if np.random.random() <= self.p:
                _[i,:] += white_noise[i,:]
        return _

    def __repr__(self):
        return self.__class__.__name__ + '(noise_std={0}, p={1})'.format(
            self.noise_std, self.p)

#%%
'Function Check'

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from torchvision import transforms
    
    def generate_sine_waves(sample_size, channel_size, seq_len):
        
        data = np.zeros((sample_size, channel_size, seq_len))
        for i in range(sample_size):
            freq = np.random.uniform(5, 7)
            t = np.linspace(0, 2 * np.pi, seq_len)
            
            for j in range(channel_size):
                amp = np.random.uniform(0.5, 1.5)
                data[i, j] = amp * np.sin(freq * t)
        
        return data
    
    pseudo_x = generate_sine_waves(16,6,200)
    pseudo_y = np.zeros((16,5))
    pseudo_x = torch.from_numpy(pseudo_x)
    pseudo_x = pseudo_x.view(-1, pseudo_x.shape[1], pseudo_x.shape[2])
    trans = transforms.Compose([
        AddWhiteNoise(std=.3, p=.5),
        AddBaselineWander(sampling_rate=50, cutoff_freq=1.1, std=1,  p=.5),
        RandomFlip(p=.3)
        ])
    
    train_set = TensorDataset(
        trans(pseudo_x), torch.from_numpy(pseudo_y))
    train_loader = DataLoader(dataset=train_set, 
                              batch_size = 8, 
                              shuffle = False)
    
    for temp,_ in train_loader:
        idx = 0
        for s in temp:
            fig = plt.figure(figsize=(16, 8))
            for j in range(s.shape[0]):
                ax = fig.add_subplot(12, 1, j + 1,
                                     xticks=[], yticks=[])
                plt.plot(s[j].cpu().numpy(), 
                         label = "augmented")
                plt.plot(pseudo_x[idx,j,:].cpu().numpy(), 
                         label = "original")
                ax.set_title(f"Channel {[j]}")
                ax.legend()
            idx += 1
        
        break