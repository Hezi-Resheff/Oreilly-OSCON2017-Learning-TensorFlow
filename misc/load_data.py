"""
The purpose of this file is to load datasets that will be used throughout the training.
---
Data is saved to disk the first time needed. Subsequently, running this script should be instantaneous.
"""
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data
from keras.applications.vgg16 import VGG16

# ---------------------------------------------------------------------------------------------------------------------
# MNIST -- hand-written digits
# ---------------------------------------------------------------------------------------------------------------------
# DATA_DIR is where the MNIST data will be downloaded to. Later, set the path accordingly to prevent an extra download.
DATA_DIR = os.path.join(os.environ["HOME"], "data") if not 'win' in sys.platform else "c:\\tmp\\data"
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

# ---------------------------------------------------------------------------------------------------------------------
# The "topless" VGG model
# ---------------------------------------------------------------------------------------------------------------------
m = VGG16(include_top=False, weights='imagenet', pooling='avg')


