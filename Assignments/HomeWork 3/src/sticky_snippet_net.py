# Sample Command: python sticky_snippet_net.py mode model_file data_folder

import sys
import tf_net as nn

MODE = sys.argv[1]
MODE_TRAIN = 'train'
MODE_FOLD = '5fold'
MODE_TEST = 'test'

MODEL_FILE = sys.argv[2]

DATA_FOLDER = sys.argv[3]


if MODE == 'train':
    nn.train(DATA_FOLDER, MODEL_FILE)
elif MODE == 'test':
    nn.test(MODEL_FILE, DATA_FOLDER)
elif MODE == '5fold':
    nn.cross_validate_and_train(DATA_FOLDER, MODEL_FILE)








