"""
Some Useful Functions and Classes
"""

import shutil
from abc import ABCMeta, abstractmethod
from threading import Lock
from sys import stdout

import numpy as np
from scipy import stats


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
                    raise KeyError('index exceeds maximum number of instances')
                return _instances[idx]
        return wrapper
    return decorator


class SimpleProgressBar:
    def __init__(self, total_len, pat='#', show_step=False, print_freq=1):
        self.len = total_len
        self.pat = pat
        self.show_step = show_step
        self.print_freq = print_freq
        self.out_stream = stdout

    def show(self, cur, desc):
        bar_len, _ = shutil.get_terminal_size()
        # The tab between desc and the progress bar should be counted.
        # And the '|'s on both ends be counted, too
        bar_len = bar_len - self.len_with_tabs(desc+'\t') - 2
        bar_len = int(bar_len*0.8)
        cur_pos = int(((cur+1)/self.len)*bar_len)
        cur_bar = '|'+self.pat*cur_pos+' '*(bar_len-cur_pos)+'|'

        disp_str = "{0}\t{1}".format(desc, cur_bar)

        # Clean
        self.write('\033[K')

        if self.show_step and (cur % self.print_freq) == 0:
            self.write(disp_str, new_line=True)
            return

        if (cur+1) < self.len:
            self.write(disp_str)
        else:
            self.write(disp_str, new_line=True)

        self.out_stream.flush()

    @staticmethod
    def len_with_tabs(s):
        return len(s.expandtabs())

    def write(self, content, new_line=False):
        end = '\n' if new_line else '\r'
        self.out_stream.write(content+end)
