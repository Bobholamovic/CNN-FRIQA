#!/bin/bash

import os
import random
import re
import json
from tqdm import tqdm
from glob import glob
from skimage import io

import torch
from ms_ssim import MS_SSIM
metric = MS_SSIM()


IMAGE_DIR = 'distorted/'
LABEL_DIR = 'images/'
FILE_DIR = './'

RATIO_TRAIN = 0.8
RATIO_VAL = 0.1
RATIO_TEST = 0.1

TYPES = (1,2,3,4)
LEVELS = (2,3,4)

N_IMGS_PER_LAB = len(TYPES)*len(LEVELS)

def gen_lists(labels):
	image_list, label_list, score_list = [], [], []
	for label in tqdm(labels):
		im_ref = io.imread(label)
		if im_ref.ndim != 3 or im_ref.shape[-1] != 3:
			continue
		label_list.extend([label]*N_IMGS_PER_LAB)
		name, ext = os.path.splitext(os.path.basename(label))
		cur_images = [
		IMAGE_DIR+'_'.join([name, str(t), str(l)])+ext
		for t in TYPES
		for l in LEVELS
	]
		image_list.extend(cur_images)
		score_list.extend([compute_score(io.imread(dst), im_ref) for dst in cur_images])

	return image_list, label_list, score_list


def compute_score(im_dst, im_ref):
	def _to_tensor(arr):
		return torch.FloatTensor(arr).transpose(-1,-2).transpose(-2,-3).unsqueeze(0)
	with torch.no_grad():
		if torch.cuda.is_available():
			score = metric(_to_tensor(im_dst).cuda(), _to_tensor(im_ref).cuda())
		else:
			score = metric(_to_tensor(im_dst), _to_tensor(im_ref))
	return float(score.cpu())

def _get(*args):
	return globals().get('_'.join(args), [])    

# Load names
label_list = glob(os.path.join(LABEL_DIR, '*.bmp'))

# Split the dataset
n_labels = len(label_list)
random.shuffle(label_list)
tr_end = int(n_labels*RATIO_TRAIN)
val_end = tr_end+int(n_labels*RATIO_VAL)

"""
train_labels = label_list[:tr_end]
val_labels = label_list[tr_end:val_end]
test_labels = label_list[val_end:]
"""

train_labels = label_list[:-20]
val_labels = label_list[-20:]
test_labels = label_list[-10:]

for ph in ('train', 'val', 'test'):
	data_zip = zip(('img', 'ref', 'score'), gen_lists(_get(ph, 'labels')))
	data_dict = dict(data_zip)
	with open('{}_data.json'.format(ph), 'w') as fp:
		json.dump(data_dict, fp)


	
		
			
	
