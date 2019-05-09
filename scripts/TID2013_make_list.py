# A script to make data lists for pytorch code
# Database: TID2013
# Date: 2018-11-6
# 
# Edited: 2019-5-7
# Change log:
#    + txt -> json
#    + mos saved as float values

import random
import json

DATA_DIR = "distorted_images/"
REF_DIR = "reference_images/"
MOS_WITH_NAMES = "./mos_with_names.txt"
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
assert((TRAIN_RATIO + TEST_RATIO) < 1)

EXCLUDE_INDICES = ()
EXCLUDE_TYPES = (15, )
TRAIN_IMAGES = './train_images.txt'
TRAIN_LABELS = './train_labels.txt'
TRAIN_MOS = './train_scores.txt'
VAL_IMAGES = './val_images.txt'
VAL_LABELS = './val_labels.txt'
VAL_MOS = './val_scores.txt'
TEST_IMAGES = './test_images.txt'
TEST_LABELS = './test_labels.txt'
TEST_MOS = './test_scores.txt'

data_list = [line.strip().split() for line in open(MOS_WITH_NAMES, 'r')]


# Split the dataset by index
N = 25 - len(EXCLUDE_INDICES)
idcs = list(range(N))
random.shuffle(idcs)
'''
train_idcs = idcs[:int(N*TRAIN_RATIO)]
val_idcs = idcs[int(N*TRAIN_RATIO):-int(N*TEST_RATIO)]
test_idcs = idcs[-int(N*TEST_RATIO):]
'''

train_idcs = idcs[:15]
val_idcs = idcs[15:20]
test_idcs = idcs[20:]

'''
test_idcs = [8, 19]
val_idcs = [6, 10, 12]
train_idcs = [n for n in idcs if n not in test_idcs+val_idcs]
'''


def _write_list_into_file(l, f):
    with open(f, "w") as h:
        for line in l:
            h.write(line)
            h.write('\n')

train_images, train_labels, train_mos = [], [], []
val_images, val_labels, val_mos = [], [], []
test_images, test_labels, test_mos = [], [], []

for mos, image in data_list:
    ref = REF_DIR + "I" + image[1:3] + '.BMP'
    img = DATA_DIR + image
    idx = int(image[1:3])
    tpe = int(image[4:6])
    if idx not in EXCLUDE_INDICES and tpe not in EXCLUDE_TYPES:
        if idx in train_idcs:
            train_images.append(img)
            train_labels.append(ref)
            train_mos.append(float(mos))
        if idx in val_idcs:
            val_images.append(img)
            val_labels.append(ref)
            val_mos.append(float(mos))
        if idx in test_idcs:
            test_images.append(img)
            test_labels.append(ref)
            test_mos.append(float(mos))            

"""
_write_list_into_file(train_images, TRAIN_IMAGES)
_write_list_into_file(train_labels, TRAIN_LABELS)
_write_list_into_file(train_mos, TRAIN_MOS)

_write_list_into_file(val_images+test_images, VAL_IMAGES)
_write_list_into_file(val_labels+test_labels, VAL_LABELS)
_write_list_into_file(val_mos+test_mos, VAL_MOS)

_write_list_into_file(test_images, TEST_IMAGES)
_write_list_into_file(test_labels, TEST_LABELS)
_write_list_into_file(test_mos, TEST_MOS)
"""

ns = vars()
for ph in ('train', 'val', 'test'):
    data_dict = dict(img=ns['{}_images'.format(ph)], ref=ns['{}_labels'.format(ph)], score=ns['{}_mos'.format(ph)])
    with open('{}_data.json'.format(ph), 'w') as fp:
        json.dump(data_dict, fp)




