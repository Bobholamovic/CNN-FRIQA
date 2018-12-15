"""
Some Useful Functions and Classes
"""

import math
from abc import ABCMeta, abstractmethod
from threading import Lock

import shutil
import numpy as np
from scipy import stats

import torch
from torch.autograd import Variable
import torch.nn.functional as F

"""
Torch version of SSIM (from Internet)
-----------------------------------------

Including functions:
    * gaussian
    * create_window
    * _ssim_map
    * _ssim
    * ssim
    * ssim_map

"""

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) * 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim_map(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    ssim_map = _ssim_map(img1, img2, window, window_size, channel)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim_map(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)


    return _ssim_map(img1, img2, window, window_size, channel)


class AverageMeter:
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


"""
Metrics for IQA performance
-----------------------------------------

Including classes:
    * Metric (base)
    * MAE
    * SROCC
    * PLCC
    * RMSE

"""

class Metric(metaclass=ABCMeta):
    def __init__(self):
        super(Metric, self).__init__()
        self.reset()
    
    def reset(self):
        self.x1 = []
        self.x2 = []

    @abstractmethod
    def _compute(self, x1, x2):
        return

    def compute(self):
        x1_array = np.array(self.x1, dtype=np.float)
        x2_array = np.array(self.x2, dtype=np.float)
        return self._compute(x1_array.ravel(), x2_array.ravel())

    def _check_type(self, x):
        return isinstance(x, (float, int, np.ndarray))

    def update(self, x1, x2):
        if self._check_type(x1) and self._check_type(x2):
            self.x1.append(x1)
            self.x2.append(x2)
        else:
            raise TypeError('Data types not supported')

class MAE(Metric):
    def __init__(self):
        super(MAE, self).__init__()

    def _compute(self, x1, x2):
        return np.sum(np.abs(x2-x1))

class SROCC(Metric):
    def __init__(self):
        super(SROCC, self).__init__()
    
    def _compute(self, x1, x2):
        return stats.spearmanr(x1, x2)[0]

class PLCC(Metric):
    def __init__(self):
        super(PLCC, self).__init__()

    def _compute(self, x1, x2):
        return stats.pearsonr(x1, x2)[0]

class RMSE(Metric):
    def __init__(self):
        super(RMSE, self).__init__()

    def _compute(self, x1, x2):
        return np.sqrt(((x2 - x1) ** 2).mean())


def limited_instances(n):
    def decorator(cls):
        _instances = [None]*n
        _lock = Lock()
        def wrapper(idx, *args, **kwargs):
            nonlocal _instances
            with _lock:
                if idx < n:
                    if _instances[idx] is None: _instances[idx] = cls(*args, **kwargs)   
                else:
                    raise ValueError('index exceeds maximum number of instances')
                return _instances[idx]
        return wrapper
    return decorator


class SimpleProgressBar:
    def __init__(self, total_len, pat='#', bar_len=50, show_step=False, print_freq=1):
        self.len = total_len
        self.pat = pat
        self.bar_len = 50
        self.show_step=show_step
        self.print_freq = print_freq

    def show(self, cur, disp_str):
        cur_bar_len = int((cur/self.len)*self.bar_len)
        cur_bar = '|'+self.pat*cur_bar_len+' '*(self.bar_len-cur_bar_len)+'|'
        
        if self.show_step:
            if (cur % self.print_freq) == 0: print("{0}\t\t{1}".format(disp_str, cur_bar), end='\n')
            return 
        if cur < self.len:
            print("{0}\t\t{1}".format(disp_str, cur_bar), end='\r')
        else:
            print("{0}\t\t{1}".format(disp_str, cur_bar), end='\n')

