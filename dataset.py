import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import butter, sosfiltfilt

# 定义滤波器类
class HighpassFilter:
    def __init__(self, fs: int, cutoff: float, order: int = 5):
        self.sos = butter(order, cutoff, btype='highpass', fs=fs, output='sos')

    def __call__(self, x):
        return sosfiltfilt(self.sos, x)

class LowpassFilter:
    def __init__(self, fs: int, cutoff: float, order: int = 5):
        self.sos = butter(order, cutoff, btype='lowpass', fs=fs, output='sos')

    def __call__(self, x):
        return sosfiltfilt(self.sos, x)



class PretrainDataset(Dataset):
    name = 'physionet'

    def __init__(self, index_path, data_standardization=True):
        self.data_standardization = data_standardization
        index_arr = []
        for line in open(index_path):
            temp = line.strip().split()
            index_arr.append(temp)
        self.index_arr = index_arr
        self.highpass_filter = HighpassFilter(fs=250, cutoff=0.67)
        self.lowpass_filter = LowpassFilter(fs=250, cutoff=40)

    
    def __len__(self):
        return len(self.index_arr)

    def __getitem__(self, index):
        data = self.index_arr[index]
        x = np.load(data[0]).astype(np.float32) 
        x = x[:, 125:-125]
        if self.data_standardization and np.std(x) != 0:
            x = (x - np.mean(x)) / np.std(x)
        x = self.highpass_filter(x)
        x = self.lowpass_filter(x)
        x = torch.tensor(x.copy(), dtype=torch.float32)
        return x
    
class PhysionetDataset(Dataset):
    """
        physionet dataset for fintune(train, validation), and test
    """
    name = 'physionet'

    def __init__(self, index_path, data_standardization=True, dataset_name=None):
        self.dataset_name = dataset_name
        self.data_standardization = data_standardization
        index_arr = []
        for line in open(index_path):
            temp = line.strip().split()
            index_arr.append(temp)
        self.index_arr = index_arr

        if dataset_name in ["p1-2_people2000_segmentall_sample_step100_data"]:
            self.label_map = {'N':0, 'Q':1, 'S':2, 'V':3}
        elif "ptb-xl" in dataset_name:
            self.label_map = {'CD':0, 'HYP':1, 'MI':2, 'NORM':3, 'STTC':4}
        elif "cpsc2018" in dataset_name:
            self.label_map = {'AF':0, 'IAVB':1, 'LBBB':2, 'NSR':3, 'PAC':4,
                              'PVC':5, 'RBBB':6, 'STD':7, 'STE':8}

        self.highpass_filter = HighpassFilter(fs=250, cutoff=0.67)
        self.lowpass_filter = LowpassFilter(fs=250, cutoff=40)

    def __len__(self):
        return len(self.index_arr)

    def __getitem__(self, index):
        data = self.index_arr[index]
        x = np.load(data[0]).astype(np.float32)
        if self.data_standardization and np.std(x) != 0:
            x = (x - np.mean(x)) / np.std(x)
        x = x[125:-125]
        x = self.highpass_filter(x)
        x = self.lowpass_filter(x)
        x = torch.tensor(x.copy(), dtype=torch.float32)

        if "ptb-xl" in self.dataset_name:
            x = x.unsqueeze(0)
            target = torch.tensor(self.label_map[data[1]], dtype=torch.long)
        elif "cpsc2018" in self.dataset_name:
            target = torch.tensor(self.label_map[data[1]], dtype=torch.long)
        else:
            x = x.unsqueeze(0)
            target = torch.tensor(self.label_map[data[2]], dtype=torch.long)
            
        return x, target