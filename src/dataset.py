"""
Dataset and Transforms
"""


import torch.utils.data
import numpy as np
import random
from skimage import io
from os.path import join, exists
from utils import limited_instances

INVALID_VALUE = -1.0

class IQADataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, ptch_size=32, n_ptchs=16, sample_once=False, list_dir=None):
        super(IQADataset, self).__init__()

        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.ptch_size = ptch_size
        self.n_ptchs = n_ptchs
        self.img_list = None
        self.ref_list = None
        self.score_list = None
        self.sample_once = sample_once
        self._from_cache = False

        self._read_lists()

        self.tfs = Transforms()
        if sample_once:
            @limited_instances(self.__len__())
            class IncrementCache:
                def store(self, data):
                    self.data = data

            self._cache = IncrementCache
            self._store_to_cache()
            self._from_cache = True

    def __getitem__(self, index):
        global INVALID_VALUE

        img = io.imread(join(self.data_dir, self.img_list[index]))
        ref = io.imread(join(self.data_dir, self.ref_list[index]))
        score = INVALID_VALUE

        if self._from_cache:
            (img_ptchs, ref_ptchs), score = self._cache(index).data
        else:
            if self.phase == 'train':
                # img, ref = self.tfs.horizontal_flip(img, ref)
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

    def _to_patch_tensors(self, img, ref):
            img_ptchs, ref_ptchs = self.tfs.to_patches(img, ref, ptch_size=self.ptch_size, n_ptchs=self.n_ptchs)
            img_ptchs, ref_ptchs = self.tfs.to_tensor(img_ptchs, ref_ptchs)
            return img_ptchs, ref_ptchs

    def _store_to_cache(self):
        for index in range(self.__len__()):
            self._cache(index).store(self.__getitem__(index))

    def _read_lists(self):
        img_path = join(self.list_dir, self.phase + '_images.txt')
        ref_path = join(self.list_dir, self.phase + '_labels.txt')
        score_path = join(self.list_dir, self.phase + '_scores.txt')

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
    """
    def __init__(self):
        super(Transforms, self).__init__()

    def _pair_deco(tf_func):
        def transform(self, img, ref=None, *args, **kwargs):
            """ image shape (w, h, c) """
            ret = tf_func(self, img, *args, **kwargs)
            img_tf = ret[0] if isinstance(ret, tuple) else ret
            if ref is None:
                return img_tf
            else:
                ## No check of key existance here
                kwargs['params'] = ret[1:]  # Add returned parameters to dict
                ref_tf = tf_func(self, ref, *args, **kwargs)
                return img_tf, ref_tf
        return transform

    def _horizontal_flip(self, img, params):
        if params is None:
            flip = (random.random() > 0.5)
            return (img[...,::-1,:] if flip else img), (flip,)
        return img[...,::-1,:] if params[0] else img

    def _to_tensor(self, img, params):
        return torch.from_numpy((img.astype(np.float32)/255).swapaxes(-3,-2).swapaxes(-3,-1))

    def _crop_square(self, img, crop_size, params):
        if params is None:
            h, w = img.shape[-3:-1]
            assert(crop_size <= h and crop_size <= w)
            ub = random.randint(0, h-crop_size)
            lb = random.randint(0, w-crop_size)
            pos = (ub, ub+crop_size, lb, lb+crop_size)
            return img[...,pos[0]:pos[1],pos[-2]:pos[-1]], pos
        return img[...,params[0]:params[1],params[-2]:params[-1]]

    def _extract_patches(self, img, ptch_size):
        h, w = img.shape[-3:-1]
        nh, nw = h//ptch_size, w//ptch_size
        assert(nh>0 and nw>0)
        vptchs = np.stack(np.split(img[...,:nh*ptch_size,:,:], nh, axis=-3))
        ptchs = np.concatenate(np.split(vptchs[...,:nw*ptch_size,:], nw, axis=-2))
        return ptchs, nh*nw

    def _to_patches(self, img, ptch_size=32, n_ptchs=None, params=None):
        ptchs, n = self._extract_patches(img, ptch_size)
        if not n_ptchs:
            n_ptchs = n
        elif n_ptchs > n:
            n_ptchs = n  
        if params is None:
            idx = list(range(n))
            random.shuffle(idx)
            idx = idx[:n_ptchs]
            return ptchs[idx], idx
        return ptchs[params]

    @_pair_deco
    def horizontal_flip(self, img, params=None):
        return self._horizontal_flip(img, params=params)

    @_pair_deco
    def to_tensor(self, img, params=None):
        return self._to_tensor(img, params=params)

    @_pair_deco
    def crop_square(self, img, crop_size=64, params=None):
        return self._crop_square(img, crop_size=crop_size, params=params)

    @_pair_deco
    def to_patches(self, img, ptch_size=32, n_ptchs=None, params=None):
        return self._to_patches(img, ptch_size=ptch_size, n_ptchs=n_ptchs, params=params)
