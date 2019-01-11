"""
Dataset and Transforms
"""


import torch.utils.data
import numpy as np
import random
from skimage import io
from os.path import join, exists
from utils import limited_instances

class IQADataset(torch.utils.data.Dataset):
    INVALID_VALUE = -1.0
    def __init__(self, data_dir, phase, ptch_size=32, n_ptchs=16, sample_once=False, \
                    subset='', list_dir=None):
        super(IQADataset, self).__init__()

        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.subset = phase if not len(subset) else subset
        self.ptch_size = ptch_size
        self.n_ptchs = n_ptchs
        self.img_list = None
        self.ref_list = None
        self.score_list = None
        self.sample_once = sample_once
        self._from_pool = False

        self._read_lists()

        self.tfs = Transforms()
        if sample_once:
            @limited_instances(self.__len__())
            class IncrementCache:
                def store(self, data):
                    self.data = data

            self._pool = IncrementCache
            self._to_pool()
            self._from_pool = True

    def __getitem__(self, index):
        img = self._loader(self.img_list[index])
        ref = self._loader(self.ref_list[index])
        score = self.INVALID_VALUE

        if self._from_pool:
            (img_ptchs, ref_ptchs), score = self._pool(index).data
        else:
            if self.phase == 'train':
                img, ref = self.tfs.horizontal_flip(img, ref)
                img_ptchs, ref_ptchs = self._to_patch_tensors(img, ref)
                score = self.score_list[index]
            elif self.phase == 'val':
                img_ptchs, ref_ptchs = self._to_patch_tensors(img, ref)
                score = self.score_list[index]
            elif self.phase == 'test':
                img_ptchs, ref_ptchs = self._to_patch_tensors(img, ref)
                if self.score_list is not None:
                    score = self.score_list[index] 
            else:
                pass

            # For TID2013
            score = (9.0 - score) / 9.0 * 100.0

        # print(img_ptchs.shape, ref_ptchs.shape, score)
        return (img_ptchs, ref_ptchs), torch.FloatTensor([score,])

    def __len__(self):
        return len(self.img_list)

    def _loader(self, iname):
        return io.imread(join(self.data_dir, iname))

    def _to_patch_tensors(self, img, ref):
            img_ptchs, ref_ptchs = self.tfs.to_patches(img, ref, ptch_size=self.ptch_size, n_ptchs=self.n_ptchs)
            img_ptchs, ref_ptchs = self.tfs.to_tensor(img_ptchs, ref_ptchs)
            return img_ptchs, ref_ptchs

    def _to_pool(self):
        for index in range(self.__len__()):
            self._pool(index).store(self.__getitem__(index))

    def _read_lists(self):
        img_path = join(self.list_dir, self.subset + '_images.txt')
        ref_path = join(self.list_dir, self.subset + '_labels.txt')
        score_path = join(self.list_dir, self.subset + '_scores.txt')

        assert exists(img_path)

        self.img_list = [line.strip() for line in open(img_path, 'r')]
        self.ref_list = [line.strip() for line in open(ref_path, 'r')]

        if exists(score_path):
            self.score_list = [float(line) for line in open(score_path, 'r')]

        if self.phase == 'train':
            extra = self.ref_list[::115]*20
            self.img_list.extend(extra)
            self.ref_list.extend(extra)

            self.img_list *= 16
            self.ref_list *= 16

            if exists(score_path):
                self.score_list.extend([9.0]*len(extra))
                self.score_list *= 16


class Transforms:
    """
    Self-designed transformation class
    ------------------------------------
    
    Several things to fix and improve:
    1. Strong couling with Dataset cuz transformation types can't 
        be simply assigned in training or testing code. (e.g. given
        a list of transforms as parameters to construct Dataset Obj)
    2. Might be unsafe in multi-thread cases
    3. Too complex decorators, not pythonic
    4. The number of params of the wrapper and the inner function should
        be the same to avoid confusion
    5. The use of params and isinstance() is not so elegant. For this, 
        consider to stipulate a fix number and type of returned values for
        inner tf functions and do all the forwarding and passing work inside
        the decorator. tf_func applies transformation, which is all it does. 
    6. Performance has not been optimized at all
    7. Doc it
    8. Supports only numpy arrays
    """
    def __init__(self):
        super(Transforms, self).__init__()

    def _pair_deco(tf_func):
        def transform(self, img, ref=None, *args, **kwargs):
            """ image shape (w, h, c) """
            if (ref is not None) and (not isinstance(ref, np.ndarray)):
                args = (ref,)+args
                ref = None
            ret = tf_func(self, img, None, *args, **kwargs)
            assert(len(ret) == 2)
            if ref is None:
                return ret[0]
            else:
                num_var = tf_func.__code__.co_argcount-3    # self, img, ref not counted
                if (len(args)+len(kwargs)) == (num_var-1): 
                    # The last parameter is special
                    # Resend it if necessary
                    var_name = tf_func.__code__.co_varnames[-1]
                    kwargs[var_name] = ret[1]
                tf_ref, _ = tf_func(self, ref, None, *args, **kwargs)
                return ret[0], tf_ref
        return transform

    def _horizontal_flip(self, img, flip):
        if flip is None:
            flip = (random.random() > 0.5)
        return (img[...,::-1,:] if flip else img), flip

    def _to_tensor(self, img):
        return torch.from_numpy((img.astype(np.float32)/255).swapaxes(-3,-2).swapaxes(-3,-1)), ()

    def _crop_square(self, img, crop_size, pos):
        if pos is None:
            h, w = img.shape[-3:-1]
            assert(crop_size <= h and crop_size <= w)
            ub = random.randint(0, h-crop_size)
            lb = random.randint(0, w-crop_size)
            pos = (ub, ub+crop_size, lb, lb+crop_size)
        return img[...,pos[0]:pos[1],pos[-2]:pos[-1]], pos

    def _extract_patches(self, img, ptch_size):
        h, w = img.shape[-3:-1]
        nh, nw = h//ptch_size, w//ptch_size
        assert(nh>0 and nw>0)
        vptchs = np.stack(np.split(img[...,:nh*ptch_size,:,:], nh, axis=-3))
        ptchs = np.concatenate(np.split(vptchs[...,:nw*ptch_size,:], nw, axis=-2))
        return ptchs, nh*nw

    def _to_patches(self, img, ptch_size, n_ptchs, idx):
        ptchs, n = self._extract_patches(img, ptch_size)
        if not n_ptchs:
            n_ptchs = n
        elif n_ptchs > n:
            n_ptchs = n  
        if idx is None:
            idx = list(range(n))
            random.shuffle(idx)
            idx = idx[:n_ptchs]
        return ptchs[idx], idx

    @_pair_deco
    def horizontal_flip(self, img, ref=None, flip=None):
        return self._horizontal_flip(img, flip=flip)

    @_pair_deco
    def to_tensor(self, img, ref=None):
        return self._to_tensor(img)

    @_pair_deco
    def crop_square(self, img, ref=None, crop_size=64, pos=None):
        return self._crop_square(img, crop_size=crop_size, pos=pos)

    @_pair_deco
    def to_patches(self, img, ref=None, ptch_size=32, n_ptchs=None, idx=None):
        return self._to_patches(img, ptch_size=ptch_size, n_ptchs=n_ptchs, idx=idx)
