#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
A CNN model to tackle the Signs dataset using tf.keras

Author: André Pacheco
Email: pacheco.comp@gmail.com

If you find any bug, please email-me
"""

from __future__ import print_function
from __future__ import division
import tensorflow as tf


import sys
#sys.path.append('/home/patcha/Dropbox/Doutorado/Codigos/Python/utils')
sys.path.append('/home/labcin/CODIGOS/utils/')
from utils_img import get_path_and_labels_from_folders, get_dataset_tf
from utils import plot_conf_matrix


