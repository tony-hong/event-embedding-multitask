'''
config.py


Author: Tony Hong

This file should be palced at the dir of the source codes "src/".
Overall structure:
	+ base
	|-- src 	# source code
	|-- corpus 	# labelled corpus
	|-- data 	# data
	|-- model 	# model of training output
'''

import os


'''
Basic configuration
'''
# Base dir
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# src dit
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Labelled corpus path
CORPUS_PATH = os.path.join(BASE_DIR, 'corpus/')

# Data path
DATA_PATH = os.path.join(BASE_DIR, 'data/')

# Model path
MODEL_PATH = os.path.join(BASE_DIR, 'model/')

# Evaluation data path
EVAL_PATH = os.path.join(BASE_DIR, 'eval_data/')

# Data path
DATA_VERSION = os.path.join(DATA_PATH, 'test/')

# Model path
MODEL_VERSION = os.path.join(MODEL_PATH, 'test/')

# Semantic role dict
ROLES_SHORT = {
    "A0": 0, 
    "A1": 1, 
    "AM-LOC": 2, 
    "AM-TMP": 3, 
    "AM-MNR": 4, 
    "V": 5,
    "<OTHER>": 6
    }

ROLES = {
    'A0': 5,
    'A1': 1,
    'A2': 4,
    'A3': 10,
    'A4': 14,
    'A5': 28,
    'AM': 25,
    'AM-ADV': 12,
    'AM-CAU': 8,
    'AM-DIR': 18,
    'AM-DIS': 6,
    'AM-EXT': 24,
    'AM-LOC': 2,
    'AM-MNR': 7,
    'AM-MOD': 21,
    'AM-NEG': 13,
    'AM-PNC': 15,
    'AM-PRD': 27,
    'AM-TMP': 3,
    'C-A1': 26,
    'C-V': 9,
    'R-A0': 11,
    'R-A1': 17,
    'R-A2': 20,
    'R-A3': 29,
    'R-AM-CAU': 23,
    'R-AM-LOC': 16,
    'R-AM-MNR': 19,
    'R-AM-TMP': 22,
    '<OTHER>': 30
    }


'''
Default Training configuration
# Multi-task role filler (MTRF)
'''

LEARNING_RATE = 0.1

# Mini-batch size for NN
BATCH_SIZE = 3000


SAMPLES_EPOCH = 100000000


EPOCHS = 200


PRINT_AFTER = 100


SAVE_AFTER = 4000


FACTOR_NUM = 256


HIDDEN_NUM = 512


LOSS_WEIGHT_ROLE = 1.0


LEARNING_RATE_DECAY = 1.0

L1_REG = 0.00

L2_REG = 0.00



'''
Data size
Version:
    October 2016
'''
OCT_VALID_SIZE = 1561092

OCT_TEST_SIZE = 1576000



'''
Data size
Version:
    March 2017
'''
# TODO need to compute
MAR_VALID_SIZE = 1561092

MAR_TEST_SIZE = 1576000



'''
Data size
Version:
    Nov 2017 processed , March 2017 version
'''
# TODO need to compute
MAR17_VALID_SIZE = 200000

MAR17_TEST_SIZE = 200000



'''
# RNN
'''
# Mini-batch size for RNN
BATCH_SIZE_RNN = 128



'''
Evaluation configuration
'''
### *** Important ***
#   This option is for evaluation of Semantic Role Classification where semantic role is not given for each input word.
#   Do not set this to True during the training! 
# SRC = True
SRC = False

