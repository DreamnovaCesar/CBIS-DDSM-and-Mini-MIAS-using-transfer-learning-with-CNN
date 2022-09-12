import os
import string
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from itertools import cycle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras import Input
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet152V2

from tensorflow.keras.applications import MobileNet

from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import MobileNetV3Large

from tensorflow.keras.applications import Xception

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet201

from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import NASNetLarge

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.applications import EfficientNetB7

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras import regularizers

from sklearn.model_selection import train_test_split

uniform_data = np.random.rand(2, 2)

Height = 12
Width = 12
Annot_kws = 12
font = 0.7

X_size_figure = 2
Y_size_figure = 2

# * Figure's size
plt.figure(figsize = (Width, Height))
plt.subplot(X_size_figure, Y_size_figure, 4)
sns.set(font_scale = font)

# * Confusion matrix heatmap
ax = sns.heatmap(uniform_data)
#ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values')

# * Subplot Training accuracy
plt.subplot(X_size_figure, Y_size_figure, 1)
plt.plot([0, 1, 2], label = 'Training Accuracy')
plt.plot([0.1, 1.1, 2.1], label = 'Validation Accuracy')
plt.ylim([0, 1])
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')

plt.subplot(X_size_figure, Y_size_figure, 2)
plt.plot([0.99, 0.22, 0.44], label = 'Training Loss')
plt.plot([0.99, 0.33, 0.55], label = 'Validation Loss')
plt.ylim([0, 2.0])
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')

plt.show()

