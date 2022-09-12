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

"""
class pretrainedModels:

  def __init__(self, **kwargs):
    
    self.folder = kwargs.get('folder', None)
    self.newfolder = kwargs.get('newfolder', None)
    self.extension = kwargs.get('extension', None)
    self.newextension = kwargs.get('newextension', None)
"""
##################################################################################################################################################################

# ? Configuration of each DCNN model

def configuration_models(All_images, All_labels, Dataframe_save, Folder_path, DL_model, Enhancement_technique, Class_labels, Column_names, X_size, Y_size, Vali_split, Epochs, Folder_data, Folder_models, Folder_models_esp):
    """
    _summary_

    _extended_summary_

    Args:
        All_images (_type_): _description_
        All_labels (_type_): _description_
        Dataframe_save (_type_): _description_
        Folder_path (_type_): _description_
        DL_model (_type_): _description_
        Enhancement_technique (_type_): _description_
        Class_labels (_type_): _description_
        Column_names (_type_): _description_
        X_size (_type_): _description_
        Y_size (_type_): _description_
        Vali_split (_type_): _description_
        Epochs (_type_): _description_
        Folder_data (_type_): _description_
        Folder_models (_type_): _description_
        Folder_models_esp (_type_): _description_

    Returns:
        _type_: _description_
    """
    #print(X_size)
    #print(Y_size)

    for Index, Model in enumerate(DL_model):

      #print(All_images)
      #print(All_labels)

      #All_images[0] = np.array(All_images[0])
      #All_images[1] = np.array(All_images[1])

      #All_labels[0] = np.array(All_labels[0])
      #All_labels[1] = np.array(All_labels[1])

      #print(len(All_images[0]))
      #print(len(All_images[1]))

      #print(len(All_labels[0]))
      #print(len(All_labels[1]))

      #All_images_CNN = All_images[0] + All_images[1]
      #All_labels_CNN = np.concatenate((All_labels[0], All_labels[1]), axis = None)

      print(len(All_images))
      print(len(All_labels))

      X_train, X_test, y_train, y_test = train_test_split(np.array(All_images), np.array(All_labels), test_size = 0.20, random_state = 42)

      Info_model = deep_learning_models(Model, Enhancement_technique, Class_labels, X_size, Y_size, Vali_split, Epochs, X_train, y_train, X_test, y_test, Folder_models, Folder_models_esp)
      
      Info_dataframe = overwrite_row_CSV(Dataframe_save, Folder_path, Info_model, Column_names, Index)

    return Info_dataframe

# ? Pretrained model configurations

def deep_learning_models(Pretrained_model_function, Enhancement_technique, Class_labels, X_size, Y_size, Vali_split, Epochs, X_train, y_train, X_test, y_test, Folder_models, Folder_models_Esp):
    """
    General configuration for each model, extracting features and printing theirs values.

    Args:
        Pretrained_model_function (_type_): _description_
        Enhancement_technique (_type_): _description_
        Class_labels (_type_): _description_
        X_size (_type_): _description_
        Y_size (_type_): _description_
        Vali_split (_type_): _description_
        Epochs (_type_): _description_
        X_train (_type_): _description_
        y_train (_type_): _description_
        X_test (_type_): _description_
        y_test (_type_): _description_
        Folder_models (_type_): _description_
        Folder_models_Esp (_type_): _description_

    Returns:
        _type_: _description_
    """

    # * Parameters plt

    Height = 12
    Width = 12
    Annot_kws = 12
    font = 0.7

    X_size_figure = 2
    Y_size_figure = 2

    # * Metrics digits

    Digits = 4

    # * List
    Info = []

    # * Class problem definition
    Class_problem = len(Class_labels)

    if Class_problem == 2:
      Class_problem_prefix = '_Biclass_'
    elif Class_problem > 2:
      Class_problem_prefix = '_Multiclass_'

    # * Training fit

    Start_training_time = time.time()

    Pretrained_model, Pretrained_model_name, Pretrained_model_name_letters = Pretrained_model_function(X_size, Y_size, Class_problem)
    Pretrained_Model_History = Pretrained_model.fit(X_train, y_train, batch_size = 32, validation_split = Vali_split, epochs = Epochs)
  
    End_training_time = time.time()

    
    # * Test evaluation

    Start_testing_time = time.time()

    Loss_Test, Accuracy_Test = Pretrained_model.evaluate(X_test, y_test, verbose = 2)

    End_testing_time = time.time()

    
    # * Total time of training and testing

    Total_training_time = End_training_time - Start_training_time 
    Total_testing_time = End_testing_time - Start_testing_time

    Pretrained_model_name_technique = str(Pretrained_model_name_letters) + '_' + str(Enhancement_technique)

    if Class_problem == 2:

      Labels_biclass_number = []

      for i in range(len(Class_labels)):
        Labels_biclass_number.append(i)

      # * Get the data from the model chosen
      y_pred = Pretrained_model.predict(X_test)
      y_pred = Pretrained_model.predict(X_test).ravel()

      # * Biclass labeling
      y_pred_class = np.where(y_pred < 0.5, 0, 1)
      
      # * Confusion Matrix
      print('Confusion Matrix')
      Confusion_matrix = confusion_matrix(y_test, y_pred_class)

      print(Confusion_matrix)
      print(classification_report(y_test, y_pred_class, target_names = Class_labels))

      # * Precision
      Precision = precision_score(y_test, y_pred_class)
      print(f"Precision: {round(Precision, Digits)}")
      print("\n")

      # * Recall
      Recall = recall_score(y_test, y_pred_class)
      print(f"Recall: {round(Recall, Digits)}")
      print("\n")

      # * F1-score
      F1_score = f1_score(y_test, y_pred_class)
      print(f"F1: {round(F1_score, Digits)}")
      print("\n")

      #print(y_pred_class)
      #print(y_test)

      #print('Confusion Matrix')
      #ConfusionM_Multiclass = confusion_matrix(y_test, y_pred_class)
      #print(ConfusionM_Multiclass)

      #Labels = ['Benign_W_C', 'Malignant']
      Confusion_matrix_dataframe = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))

      # * Figure's size
      plt.figure(figsize = (Width, Height))
      plt.subplot(X_size_figure, Y_size_figure, 4)
      sns.set(font_scale = font)

      # * Confusion matrix heatmap
      ax = sns.heatmap(Confusion_matrix_dataframe, annot = True, fmt = 'd')
      #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
      ax.set_xlabel('\nPredicted Values')
      ax.set_ylabel('Actual Values')
      ax.set_xticklabels(Class_labels)
      ax.set_yticklabels(Class_labels)

      Accuracy = Pretrained_Model_History.history['accuracy']
      Validation_accuracy = Pretrained_Model_History.history['val_accuracy']

      Loss = Pretrained_Model_History.history['loss']
      Validation_loss = Pretrained_Model_History.history['val_loss']

      # * FPR and TPR values for the ROC curve
      FPR, TPR, _ = roc_curve(y_test, y_pred)
      Auc = auc(FPR, TPR)

      # * Subplot Training accuracy
      plt.subplot(X_size_figure, Y_size_figure, 1)
      plt.plot(Accuracy, label = 'Training Accuracy')
      plt.plot(Validation_accuracy, label = 'Validation Accuracy')
      plt.ylim([0, 1])
      plt.legend(loc = 'lower right')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('epoch')

      # * Subplot Training loss
      plt.subplot(X_size_figure, Y_size_figure, 2)
      plt.plot(Loss, label = 'Training Loss')
      plt.plot(Validation_loss, label = 'Validation Loss')
      plt.ylim([0, 2.0])
      plt.legend(loc = 'upper right')
      plt.title('Training and Validation Loss')
      plt.xlabel('epoch')

      # * Subplot ROC curve
      plt.subplot(X_size_figure, Y_size_figure, 3)
      plt.plot([0, 1], [0, 1], 'k--')
      plt.plot(FPR, TPR, label = Pretrained_model_name + '(area = {:.4f})'.format(Auc))
      plt.xlabel('False positive rate')
      plt.ylabel('True positive rate')
      plt.title('ROC curve')
      plt.legend(loc = 'lower right')

      # * Save this figure in the folder given
      Class_problem_name = str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.png'
      Class_problem_folder = os.path.join(Folder_models, Class_problem_name)

      plt.savefig(Class_problem_folder)
      #plt.show()

    elif Class_problem >= 3:
    
      Labels_multiclass_number = []

      for i in range(len(Class_labels)):
        Labels_multiclass_number.append(i)

      # * Get the data from the model chosen
      y_pred = Pretrained_model.predict(X_test)
      y_pred = np.argmax(y_pred, axis = 1)

      # * Multiclass labeling
      y_pred_roc = label_binarize(y_pred, classes = Labels_multiclass_number)
      y_test_roc = label_binarize(y_test, classes = Labels_multiclass_number)

      #print(y_pred)
      #print(y_test)

      # * Confusion Matrix
      print('Confusion Matrix')
      Confusion_matrix = confusion_matrix(y_test, y_pred)

      print(Confusion_matrix)
      print(classification_report(y_test, y_pred, target_names = Class_labels))

      # * Precision
      Precision = precision_score(y_test, y_pred, average = 'weighted')
      print(f"Precision: {round(Precision, Digits)}")
      print("\n")

      # * Recall
      Recall = recall_score(y_test, y_pred, average = 'weighted')
      print(f"Recall: {round(Recall, Digits)}")
      print("\n")

      # * F1-score
      F1_score = f1_score(y_test, y_pred, average = 'weighted')
      print(f"F1: {round(F1_score, Digits)}")
      print("\n")

      #labels = ['Benign', 'Benign_W_C', 'Malignant']
      df_cm = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))

      plt.figure(figsize = (Width, Height))
      plt.subplot(X_size_figure, Y_size_figure, 4)
      sns.set(font_scale = font) # for label size

      ax = sns.heatmap(df_cm, annot = True, fmt = 'd', annot_kws = {"size": Annot_kws}) # font size
      #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
      ax.set_xlabel('\nPredicted Values')
      ax.set_ylabel('Actual Values ')
      ax.set_xticklabels(Class_labels)
      ax.set_yticklabels(Class_labels)

      Accuracy = Pretrained_Model_History.history['accuracy']
      Validation_Accuracy = Pretrained_Model_History.history['val_accuracy']

      Loss = Pretrained_Model_History.history['loss']
      Validation_Loss = Pretrained_Model_History.history['val_loss']

      FPR = dict()
      TPR = dict()
      Roc_auc = dict()

      for i in range(Class_problem):
        FPR[i], TPR[i], _ = roc_curve(y_test_roc[:, i], y_pred_roc[:, i])
        Roc_auc[i] = auc(FPR[i], TPR[i])

      # * Colors for ROC curves
      Colors = cycle(['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan'])
      
      # * Subplot Training accuracy
      plt.subplot(X_size_figure, Y_size_figure, 1)
      plt.plot(Accuracy, label = 'Training Accuracy')
      plt.plot(Validation_Accuracy, label = 'Validation Accuracy')
      plt.ylim([0, 1])
      plt.legend(loc = 'lower right')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('epoch')

      # * Subplot Training loss
      plt.subplot(X_size_figure, Y_size_figure, 2)
      plt.plot(Loss, label = 'Training Loss')
      plt.plot(Validation_Loss, label = 'Validation Loss')
      plt.ylim([0, 2.0])
      plt.legend(loc = 'upper right')
      plt.title('Training and Validation Loss')
      plt.xlabel('epoch')

      # * Subplot ROC curves
      plt.subplot(X_size_figure, Y_size_figure, 3)
      plt.plot([0, 1], [0, 1], 'k--')

      for i, color, lbl in zip(range(Class_problem), Colors, Class_labels):
        plt.plot(FPR[i], TPR[i], color = color, label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(lbl, Roc_auc[i]))

      plt.legend(loc = 'lower right')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False positive rate')
      plt.ylabel('True positive rate')
      plt.title('ROC curve Multiclass')

      # * Save this figure in the folder given
      Class_problem_name = str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.png'
      Class_problem_folder = os.path.join(Folder_models, Class_problem_name)

      plt.savefig(Class_problem_folder)
      #plt.show()

    Info.append(Pretrained_model_name_technique)
    Info.append(Pretrained_model_name)
    Info.append(Accuracy[Epochs - 1])
    Info.append(Accuracy[0])
    Info.append(Accuracy_Test)
    Info.append(Loss[Epochs - 1])
    Info.append(Loss_Test)
    Info.append(len(y_train))
    Info.append(len(y_test))
    Info.append(Precision)
    Info.append(Recall)
    Info.append(F1_score)
    Info.append(Total_training_time)
    Info.append(Total_testing_time)
    Info.append(Enhancement_technique)
    Info.append(Confusion_matrix[0][0])
    Info.append(Confusion_matrix[0][1])
    Info.append(Confusion_matrix[1][0])
    Info.append(Confusion_matrix[1][1])
    Info.append(Epochs)
  
    if Class_problem == 2:
      Info.append(Auc)
    elif Class_problem > 2:
      for i in range(Class_problem):
        Info.append(Roc_auc[i])
    
    return Info

# ? Update CSV changing value

def overwrite_row_CSV(Dataframe, Folder_path, Info_list, Column_names, Row):

    """
    Updates final CSV dataframe to see all values

    Parameters:
    argument1 (list): All values.
    argument2 (dataframe): dataframe that will be updated
    argument3 (list): Names of each column
    argument4 (folder): Folder path to save the dataframe
    argument5 (int): The index.

    Returns:
    void
    
    """

    for i in range(len(Info_list)):
        Dataframe.loc[Row, Column_names[i]] = Info_list[i]

    Dataframe.to_csv(Folder_path, index = False)

    print(Dataframe)

    return Dataframe

##################################################################################################################################################################

# ? Folder Configuration of each DCNN model

def configuration_models_folder(Training_data, Validation_data, Test_data, Dataframe_save, Folder_path, DL_model, Enhancement_technique, Class_labels, Column_names, X_size, Y_size, Epochs, Folder_CSV, Folder_models, Folder_models_esp):

    for Index, Model in enumerate(DL_model):

      Info_model = deep_learning_models_folder(Training_data, Validation_data, Test_data, Model, Enhancement_technique, Class_labels, X_size, Y_size, Epochs, Folder_CSV, Folder_models, Folder_models_esp)
      
      Dataframe_updated = overwrite_row_CSV_folder(Dataframe_save, Folder_path, Info_model, Column_names, Index)

    return Dataframe_updated

# ? Folder Pretrained model configurations

def deep_learning_models_folder(Train_data, Valid_data, Test_data, Pretrained_model_function, Enhancement_technique, Class_labels, X_size, Y_size, Epochs, Folder_CSV, Folder_models, Folder_models_Esp):

    """
	  General configuration for each model, extracting features and printing theirs values.

    Parameters:
    argument1 (model): Model chosen.
    argument2 (str): technique used.
    argument3 (list): labels used for printing.
    argument4 (int): Size of X.
    argument5 (int): Size of Y.
    argument6 (int): Number of classes.
    argument7 (float): Validation split value.
    argument8 (int): Number of epochs.
    argument9 (int): X train split data.
    argument9 (int): y train split data.
    argument9 (int): X test split data.
    argument9 (int): y test split data.
    argument9 (int): Folder used to save data images.
    argument9 (int): Folder used to save data images in spanish.

    Returns:
	  int:Returning all metadata from each model.
    
   	"""

    # * Parameters plt

    Height = 12
    Width = 12
    Annot_kws = 12
    font = 0.7

    X_size_figure = 2
    Y_size_figure = 2

    # * Parameters dic classification report

    Macro_avg_label = 'macro avg'
    Weighted_avg_label = 'weighted avg'

    Classification_report_labels = []
    Classification_report_metrics_labels = ('precision', 'recall', 'f1-score', 'support')

    for Label in Class_labels:
      Classification_report_labels.append(Label)
    
    Classification_report_labels.append(Macro_avg_label)
    Classification_report_labels.append(Weighted_avg_label)

    Classification_report_values = []

    #Precision_label = 'precision'
    #Recall_label = 'recall'
    #F1_score_label = 'f1-score'
    #Images_support_label = 'support'

    # * Metrics digits

    Digits = 4

    # * List
    Info = []

    # * Class problem definition
    Class_problem = len(Class_labels)

    if Class_problem == 2:
      Class_problem_prefix = 'Biclass_'
    elif Class_problem > 2:
      Class_problem_prefix = 'Multiclass_'

    # * Training fit

    Start_training_time = time.time()

    Pretrained_model, Pretrained_model_name, Pretrained_model_name_letters = Pretrained_model_function(X_size, Y_size, Class_problem)

    Pretrained_Model_History = Pretrained_model.fit(  Train_data,
                                                      validation_data = Valid_data,
                                                      steps_per_epoch = Train_data.n//Train_data.batch_size,
                                                      validation_steps = Valid_data.n//Valid_data.batch_size,
                                                      epochs = Epochs)

    #steps_per_epoch = Train_data.n//Train_data.batch_size,

    End_training_time = time.time()

    
    # * Test evaluation

    Start_testing_time = time.time()

    Loss_Test, Accuracy_Test = Pretrained_model.evaluate(Test_data)

    End_testing_time = time.time()

    
    # * Total time of training and testing

    Total_training_time = End_training_time - Start_training_time 
    Total_testing_time = End_testing_time - Start_testing_time

    Pretrained_model_name_technique = str(Pretrained_model_name_letters) + '_' + str(Enhancement_technique)

    if Class_problem == 2:

      Labels_biclass_number = []

      for i in range(len(Class_labels)):
        Labels_biclass_number.append(i)

      # * Get the data from the model chosen

      Predict = Pretrained_model.predict(Test_data)
      y_pred = Pretrained_model.predict(Test_data).ravel()

      print(Test_data.classes)
      print(y_pred)
      
      #y_pred = Pretrained_model.predict(X_test)
      #y_pred = Pretrained_model.predict(X_test).ravel()
      
      # * Biclass labeling
      y_pred_class = np.where(y_pred < 0.5, 0, 1)
      
      # * Confusion Matrix
      print('Confusion Matrix')
      Confusion_matrix = confusion_matrix(Test_data.classes, y_pred_class)
      
      print(Confusion_matrix)
      print(classification_report(Test_data.classes, y_pred_class, target_names = Class_labels))
      
      Report = classification_report(Test_data.classes, y_pred_class, target_names = Class_labels)
      Dict = classification_report(Test_data.classes, y_pred_class, target_names = Class_labels, output_dict = True)
      
      for i, Report_labels in enumerate(Classification_report_labels):
        for i, Metric_labels in enumerate(Classification_report_metrics_labels):

          print(Dict[Report_labels][Metric_labels])
          Classification_report_values.append(Dict[Report_labels][Metric_labels])
          print("\n")

      # * Precision
      Precision = precision_score(Test_data.classes, y_pred_class)
      print(f"Precision: {round(Precision, Digits)}")
      print("\n")

      # * Recall
      Recall = recall_score(Test_data.classes, y_pred_class)
      print(f"Recall: {round(Recall, Digits)}")
      print("\n")

      # * F1-score
      F1_score = f1_score(Test_data.classes, y_pred_class)
      print(f"F1: {round(F1_score, Digits)}")
      print("\n")

      #print(y_pred_class)
      #print(y_test)

      #print('Confusion Matrix')
      #ConfusionM_Multiclass = confusion_matrix(y_test, y_pred_class)
      #print(ConfusionM_Multiclass)

      #Labels = ['Benign_W_C', 'Malignant']
      
      Confusion_matrix_dataframe = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))

      Confusion_matrix_dataframe_name = 'Dataframe_' + str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.csv'
      Confusion_matrix_dataframe_folder = os.path.join(Folder_CSV, Confusion_matrix_dataframe_name)

      # * Figure's size
      plt.figure(figsize = (Width, Height))
      plt.subplot(X_size_figure, Y_size_figure, 4)
      sns.set(font_scale = font)

      # * Confusion matrix heatmap
      ax = sns.heatmap(Confusion_matrix_dataframe, annot = True, fmt = 'd')
      #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
      ax.set_xlabel('\nPredicted Values')
      ax.set_ylabel('Actual Values')
      ax.set_xticklabels(Class_labels)
      ax.set_yticklabels(Class_labels)

      Accuracy = Pretrained_Model_History.history['accuracy']
      Validation_accuracy = Pretrained_Model_History.history['val_accuracy']

      Loss = Pretrained_Model_History.history['loss']
      Validation_loss = Pretrained_Model_History.history['val_loss']

      # * FPR and TPR values for the ROC curve
      FPR, TPR, _ = roc_curve(Test_data.classes, y_pred_class)
      Auc = auc(FPR, TPR)

      # * Subplot Training accuracy
      plt.subplot(X_size_figure, Y_size_figure, 1)
      plt.plot(Accuracy, label = 'Training Accuracy')
      plt.plot(Validation_accuracy, label = 'Validation Accuracy')
      plt.ylim([0, 1])
      plt.legend(loc = 'lower right')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('epoch')

      # * Subplot Training loss
      plt.subplot(X_size_figure, Y_size_figure, 2)
      plt.plot(Loss, label = 'Training Loss')
      plt.plot(Validation_loss, label = 'Validation Loss')
      plt.ylim([0, 2.0])
      plt.legend(loc = 'upper right')
      plt.title('Training and Validation Loss')
      plt.xlabel('epoch')

      # * Subplot ROC curve
      plt.subplot(X_size_figure, Y_size_figure, 3)
      plt.plot([0, 1], [0, 1], 'k--')
      plt.plot(FPR, TPR, label = Pretrained_model_name + '(area = {:.4f})'.format(Auc))
      plt.xlabel('False positive rate')
      plt.ylabel('True positive rate')
      plt.title('ROC curve')
      plt.legend(loc = 'lower right')

      # * Save this figure in the folder given
      Class_problem_name = str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.png'
      Class_problem_folder = os.path.join(Folder_models, Class_problem_name)

      plt.savefig(Class_problem_folder)
      #plt.show()

    elif Class_problem >= 3:
    
      Labels_multiclass_number = []

      for i in range(len(Class_labels)):
        Labels_multiclass_number.append(i)

      # * Get the data from the model chosen
      Predict = Pretrained_model.predict(Test_data)
      y_pred = Predict.argmax(axis = 1)

      # * Multiclass labeling
      y_pred_roc = label_binarize(y_pred, classes = Labels_multiclass_number)
      y_test_roc = label_binarize(Test_data.classes, classes = Labels_multiclass_number)

      #print(y_pred)
      #print(y_test)

      # * Confusion Matrix
      print('Confusion Matrix')
      Confusion_matrix = confusion_matrix(Test_data.classes, y_pred)

      print(Confusion_matrix)
      print(classification_report(Test_data.classes, y_pred, target_names = Class_labels))
      
      #Report = classification_report(Test_data.classes, y_pred, target_names = Class_labels)
      Dict = classification_report(Test_data.classes, y_pred, target_names = Class_labels, output_dict = True)

      for i, Report_labels in enumerate(Classification_report_labels):
        for i, Metric_labels in enumerate(Classification_report_metrics_labels):

          print(Dict[Report_labels][Metric_labels])
          Classification_report_values.append(Dict[Report_labels][Metric_labels])
          print("\n")

      # * Precision
      Precision = precision_score(Test_data.classes, y_pred, average = 'weighted')
      print(f"Precision: {round(Precision, Digits)}")
      print("\n")

      # * Recall
      Recall = recall_score(Test_data.classes, y_pred, average = 'weighted')
      print(f"Recall: {round(Recall, Digits)}")
      print("\n")

      # * F1-score
      F1_score = f1_score(Test_data.classes, y_pred, average = 'weighted')
      print(f"F1: {round(F1_score, Digits)}")
      print("\n")

      #labels = ['Benign', 'Benign_W_C', 'Malignant']
      df_cm = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))

      plt.figure(figsize = (Width, Height))
      plt.subplot(X_size_figure, Y_size_figure, 4)
      sns.set(font_scale = font) # for label size

      ax = sns.heatmap(df_cm, annot = True, fmt = 'd', annot_kws = {"size": Annot_kws}) # font size
      #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
      ax.set_xlabel('\nPredicted Values')
      ax.set_ylabel('Actual Values ')
      ax.set_xticklabels(Class_labels)
      ax.set_yticklabels(Class_labels)

      Accuracy = Pretrained_Model_History.history['accuracy']
      Validation_Accuracy = Pretrained_Model_History.history['val_accuracy']

      Loss = Pretrained_Model_History.history['loss']
      Validation_Loss = Pretrained_Model_History.history['val_loss']

      FPR = dict()
      TPR = dict()
      Roc_auc = dict()

      for i in range(Class_problem):
        FPR[i], TPR[i], _ = roc_curve(y_test_roc[:, i], y_pred_roc[:, i])
        Roc_auc[i] = auc(FPR[i], TPR[i])

      # * Colors for ROC curves
      Colors = cycle(['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan'])
      
      # * Subplot Training accuracy
      plt.subplot(X_size_figure, Y_size_figure, 1)
      plt.plot(Accuracy, label = 'Training Accuracy')
      plt.plot(Validation_Accuracy, label = 'Validation Accuracy')
      plt.ylim([0, 1])
      plt.legend(loc = 'lower right')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('epoch')

      # * Subplot Training loss
      plt.subplot(X_size_figure, Y_size_figure, 2)
      plt.plot(Loss, label = 'Training Loss')
      plt.plot(Validation_Loss, label = 'Validation Loss')
      plt.ylim([0, 2.0])
      plt.legend(loc = 'upper right')
      plt.title('Training and Validation Loss')
      plt.xlabel('epoch')

      # * Subplot ROC curves
      plt.subplot(X_size_figure, Y_size_figure, 3)
      plt.plot([0, 1], [0, 1], 'k--')

      for i, color, lbl in zip(range(Class_problem), Colors, Class_labels):
        plt.plot(FPR[i], TPR[i], color = color, label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(lbl, Roc_auc[i]))

      plt.legend(loc = 'lower right')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False positive rate')
      plt.ylabel('True positive rate')
      plt.title('ROC curve Multiclass')

      # * Save this figure in the folder given
      Class_problem_name = str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.png'
      Class_problem_folder = os.path.join(Folder_models, Class_problem_name)

      plt.savefig(Class_problem_folder)
      #plt.show()
    
    Info.append(Pretrained_model_name_technique)
    Info.append(Pretrained_model_name)
    Info.append(Accuracy[Epochs - 1])
    Info.append(Accuracy[0])
    Info.append(Accuracy_Test)
    Info.append(Loss[Epochs - 1])
    Info.append(Loss_Test)
    Info.append(len(Train_data.classes))
    Info.append(len(Valid_data.classes))
    Info.append(len(Test_data.classes))
    Info.append(Precision)
    Info.append(Recall)
    Info.append(F1_score)

    for i, value in enumerate(Classification_report_values):
      Info.append(value)

    Info.append(Total_training_time)
    Info.append(Total_testing_time)
    Info.append(Enhancement_technique)
    Info.append(Confusion_matrix[0][0])
    Info.append(Confusion_matrix[0][1])
    Info.append(Confusion_matrix[1][0])
    Info.append(Confusion_matrix[1][1])
    Info.append(Epochs)
  
    if Class_problem == 2:
      Info.append(Auc)
    elif Class_problem > 2:
      for i in range(Class_problem):
        Info.append(Roc_auc[i])
    
    return Info

# ? Folder Update CSV changing value

def overwrite_row_CSV_folder(Dataframe, Folder_path, Info_list, Column_names, Row):

    """
	  Updates final CSV dataframe to see all values

    Parameters:
    argument1 (list): All values.
    argument2 (dataframe): dataframe that will be updated
    argument3 (list): Names of each column
    argument4 (folder): Folder path to save the dataframe
    argument5 (int): The index.

    Returns:
	  void
    
   	"""

    for i in range(len(Info_list)):
        Dataframe.loc[Row, Column_names[i]] = Info_list[i]
  
    Dataframe.to_csv(Folder_path, index = False)
  
    print(Dataframe)

    return Dataframe

##################################################################################################################################################################

# ? Fine-Tuning MLP

def MLP_classificador(x, Units: int, Activation: string):
  """
  MLP configuration.

  Args:
      x (list): Layers.
      Units (int): The number of units for last layer.
      Activation (string): Activation used.

  Returns:
      _type_: _description_
  """
  x = Flatten()(x)
  x = BatchNormalization()(x)
  x = Dropout(0.6)(x)
  x = Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.01))(x)
  x = BatchNormalization()(x)
  x = Dropout(0.6)(x)
  #x = BatchNormalization()(x)
  x = Dense(Units, activation = Activation)(x)

  return x

##################################################################################################################################################################

# ? Model function

def Model_pretrained(X_size: int, Y_size: int, Num_classes: int, Model_pretrained_value: int):
    """
    Model configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.
        Model_pretrained_value (int): Choses the model.

    Returns:
        _type_: _description_
        string: Returning Model.
        string: Returning Model Name.
    """

    def model_pretrained_index(Model_pretrained_value: int):

        if (Model_pretrained_value == 1):

            Model_name = 'EfficientNetB7_Model'
            Model_name_letters = 'ENB7'
            Model_index_chosen = EfficientNetB7

            return Model_name, Model_name_letters, Model_index_chosen

        if (Model_pretrained_value == 2):

            Model_name = 'EfficientNetB6_Model'
            Model_name_letters = 'ENB6'
            Model_index_chosen = EfficientNetB6

            return Model_name, Model_name_letters, Model_index_chosen

        if (Model_pretrained_value == 3):

            Model_name = 'EfficientNetB5_Model'
            Model_name_letters = 'ENB5'
            Model_index_chosen = EfficientNetB5

            return Model_name, Model_name_letters, Model_index_chosen

        if (Model_pretrained_value == 4):

            Model_name = 'EfficientNetB4_Model'
            Model_name_letters = 'ENB4'
            Model_index_chosen = EfficientNetB4

            return Model_name, Model_name_letters, Model_index_chosen

        if (Model_pretrained_value == 5):
            
            Model_name = 'EfficientNetB3_Model'
            Model_name_letters = 'ENB3'
            Model_index_chosen = EfficientNetB3

            return Model_name, Model_name_letters, Model_index_chosen

        if (Model_pretrained_value == 6):
            
            Model_name = 'EfficientNetB2_Model'
            Model_name_letters = 'ENB2'
            Model_index_chosen = EfficientNetB2

            return Model_name, Model_name_letters, Model_index_chosen

        if (Model_pretrained_value == 7):
            
            Model_name = 'EfficientNetB1_Model'
            Model_name_letters = 'ENB1'
            Model_index_chosen = EfficientNetB1

            return Model_name, Model_name_letters, Model_index_chosen

        if (Model_pretrained_value == 8):
            
            Model_name = 'EfficientNetB0_Model'
            Model_name_letters = 'ENB0'
            Model_index_chosen = EfficientNetB0

            return Model_name, Model_name_letters, Model_index_chosen
        
        if (Model_pretrained_value == 9):
            
            Model_name = 'ResNet50_Model'
            Model_name_letters = 'RN50'
            Model_index_chosen = ResNet50

            return Model_name, Model_name_letters, Model_index_chosen
        
        if (Model_pretrained_value == 10):
            
            Model_name = 'ResNet50V2_Model'
            Model_name_letters = 'RN50V2'
            Model_index_chosen = ResNet50V2

            return Model_name, Model_name_letters, Model_index_chosen

        if (Model_pretrained_value == 11):
            
            Model_name = 'ResNet152_Model'
            Model_name_letters = 'RN152'
            Model_index_chosen = ResNet152

            return Model_name, Model_name_letters, Model_index_chosen

        if (Model_pretrained_value == 12):
            
            Model_name = 'ResNet152V2_Model'
            Model_name_letters = 'RN152V2'
            Model_index_chosen = ResNet152V2    

        if (Model_pretrained_value == 12):
            
            Model_name = 'ResNet152V2_Model'
            Model_name_letters = 'RN152V2'
            Model_index_chosen = ResNet152V2     

            return Model_name, Model_name_letters, Model_index_chosen

        return -1
        
    Model_name, Model_name_letters, Model_index_chosen = model_pretrained_index(Model_pretrained_value: int)

    Model_input = Model_index_chosen(  input_shape = (X_size, Y_size, 3), 
                                                include_top = False, 
                                                weights = "imagenet")

    for layer in Model_input.layers:
        layer.trainable = False

    if Num_classes == 2:
        Activation = 'sigmoid'
        Loss = "binary_crossentropy"
        Units = 1
    else:
        Activation = 'softmax'
        Units = Num_classes
        Loss = "categorical_crossentropy"

    x = MLP_classificador(Model_input.output, Units, Activation)

    Model_CNN = Model(Model_input.input, x)

    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    Model_CNN.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ['accuracy']
    )
  
    return Model_CNN, Model_name, Model_name_letters

# ? EfficientNetB7

def EfficientNetB7_pretrained(X_size: int, Y_size: int, Num_classes: int, Model_pretrained: int):
  """
  EfficientNetB7 configuration.

  Args:
      X_size (int): X's size value.
      Y_size (int): Y's size value.
      Num_classes (int): Number total of classes.

  Returns:
      _type_: _description_
      string: Returning EfficientNetB7 model.
      string: Returning EfficientNetB7 Name.
  """

  Model_name = 'EfficientNetB7_Model'
  Model_name_letters = 'ENB7'

  EfficientNetB7_Model = EfficientNetB7( input_shape = (X_size, Y_size, 3), 
                                                    include_top = False, 
                                                    weights = "imagenet")

  for layer in EfficientNetB7_Model.layers:
    layer.trainable = False

  if Num_classes == 2:
    Activation = 'sigmoid'
    Loss = "binary_crossentropy"
    Units = 1
  else:
    Activation = 'softmax'
    Units = Num_classes
    Loss = "categorical_crossentropy"

  
  x = MLP_classificador(EfficientNetB7_Model.output, Units, Activation)

  EfficientNetB7_model = Model(EfficientNetB7_Model.input, x)

  Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

  EfficientNetB7_model.compile(
      optimizer = Opt,
      loss = Loss,
      metrics = ['accuracy']
  )
  
  return EfficientNetB7_model, Model_name, Model_name_letters

# ? ResNet50

def ResNet50_pretrained(X_size: int, Y_size: int, Num_classes: int):
  """
  ResNet50 configuration.

  Args:
      X_size (int): X's size value.
      Y_size (int): Y's size value.
      Num_classes (int): Number total of classes.

  Returns:
      _type_: _description_
      string: Returning ResNet50 model.
      string: Returning ResNet50 Name.
  """

  Model_name = 'ResNet50_Model'
  Model_name_letters = 'RN50'

  ResNet50_Model = ResNet50(input_shape = (X_size, Y_size, 3), 
                            include_top = False, 
                            weights = "imagenet")

  for layer in ResNet50_Model.layers:
    layer.trainable = False

  if Num_classes == 2:
    Activation = 'sigmoid'
    Loss = "binary_crossentropy"
    Units = 1
  else:
    Activation = 'softmax'
    Units = Num_classes
    Loss = "categorical_crossentropy"

  
  x = MLP_classificador(ResNet50_Model.output, Units, Activation)

  ResNet50_model = Model(ResNet50_Model.input, x)

  Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

  ResNet50_model.compile(
      optimizer = Opt,
      loss = Loss,
      metrics = ['accuracy']
  )

  return ResNet50_model, Model_name, Model_name_letters

# ? ResNet50V2

def ResNet50V2_pretrained(X_size: int, Y_size: int, Num_classes: int):
    """
    ResNet50V2 configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning ResNet50V2 model.
        string: Returning ResNet50V2 Name.
    """

    Model_name = 'ResNet50V2_Model'
    Model_name_letters = 'RN50V2'

    ResNet50V2_Model = ResNet50V2(input_shape = (X_size, Y_size, 3), 
                                  include_top = False, 
                                  weights = "imagenet")

    for layer in ResNet50V2_Model.layers:
      layer.trainable = False
  
    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    x = MLP_classificador(ResNet50V2_Model.output, Units, Activation)

    ResNet50V2_model = Model(ResNet50V2_Model.input, x)

    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    ResNet50V2_model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ['accuracy']
    )

    return ResNet50V2_model, Model_name, Model_name_letters

# ? ResNet152

def ResNet152_pretrained(X_size: int, Y_size: int, Num_classes: int):
    """
    ResNet152 configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning ResNet152 model.
        string: Returning ResNet152 Name.
    """

    Model_name = 'ResNet152_Model'
    Model_name_letters = 'RN152'

    ResNet152_Model = ResNet152(input_shape = (X_size, Y_size, 3), 
                              include_top = False, 
                              weights = "imagenet")

    for layer in ResNet152_Model.layers:
      layer.trainable = False

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    x = MLP_classificador(ResNet152_Model.output, Units, Activation)

    ResNet152Model = Model(ResNet152_Model.input, x)

    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    ResNet152Model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ['accuracy']
    )

    return ResNet152Model, Model_name, Model_name_letters

# ? ResNet152V2

def ResNet152V2_pretrained(X_size: int, Y_size: int, Num_classes: int):
    """
    ResNet152V2 configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning ResNet152V2 model.
        string: Returning ResNet152V2 Name.
    """
     
    Model_name = 'ResNet152V2_Model'
    Model_name_letters = 'RN152V2'

    ResNet152V2_Model = ResNet152V2(input_shape = (X_size, Y_size, 3), 
                                    include_top = False, 
                                    weights = "imagenet")

    for layer in ResNet152V2_Model.layers:
      layer.trainable = False

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    x = MLP_classificador(ResNet152V2_Model.output, Units, Activation)

    ResNet152V2_model = Model(ResNet152V2_Model.input, x)

    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    ResNet152V2_model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ['accuracy']
    )

    return ResNet152V2_model, Model_name, Model_name_letters

# ? MobileNet

def MobileNet_pretrained(X_size: int, Y_size: int, Num_classes: int):
    """
    MobileNet configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning MobileNet model.
        string: Returning MobileNet Name.
    """

    Model_name = 'MobileNet_Model'
    Model_name_letters = 'MN'

    MobileNet_Model = MobileNet(input_shape = (X_size, Y_size, 3), 
                                              include_top = False, 
                                              weights = "imagenet")

    for layer in MobileNet_Model.layers:
      layer.trainable = False

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    x = MLP_classificador(MobileNet_Model.output, Units, Activation)

    MobileNet_model = Model(MobileNet_Model.input, x)

    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    MobileNet_model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ["accuracy"]
    )

    return MobileNet_model, Model_name, Model_name_letters

# ? MobileNetV3Small

def MobileNetV3Small_pretrained(X_size: int, Y_size: int, Num_classes: int):
    """
    MobileNetV3Small configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning MobileNetV3Small model.
        string: Returning MobileNetV3Small Name.
    """

    Model_name = 'MobileNetV3Small_Model'
    Model_name_letters = 'MNV3S'

    MobileNetV3Small_Model = MobileNetV3Small(input_shape = (X_size, Y_size, 3), 
                                              include_top = False, 
                                              weights = "imagenet")

    for layer in MobileNetV3Small_Model.layers:
      layer.trainable = False

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    x = MLP_classificador(MobileNetV3Small_Model.output, Units, Activation)

    MobileNetV3Small_model = Model(MobileNetV3Small_Model.input, x)

    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    MobileNetV3Small_model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ["accuracy"]
    )

    return MobileNetV3Small_model, Model_name, Model_name_letters

# ? MobileNetV3Large

def MobileNetV3Large_pretrained(X_size: int, Y_size: int, Num_classes: int):
    """
    MobileNetV3Large configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning MobileNetV3Large model.
        string: Returning MobileNetV3Large Name.
    """

    Model_name = 'MobileNetV3Large_Model'
    Model_name_letters = 'MNV3L'

    MobileNetV3Large_Model = MobileNetV3Large(input_shape = (X_size, Y_size, 3), 
                                              include_top = False, 
                                              weights = "imagenet")

    for layer in MobileNetV3Large_Model.layers:
      layer.trainable = False

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"
      
    x = MLP_classificador(MobileNetV3Large_Model.output, Units, Activation)

    MobileNetV3Large_model = Model(MobileNetV3Large_Model.input, x)

    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    MobileNetV3Large_model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ["accuracy"]
    )

    return MobileNetV3Large_model, Model_name, Model_name_letters

# ? Xception

def Xception_pretrained(X_size: int, Y_size: int, Num_classes: int):
    """
    Xception configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning Xception model.
        string: Returning Xception Name.
    """

    Model_name = 'Xception_Model'
    Model_name_letters = 'Xc'

    Xception_Model = Xception(input_shape = (X_size, Y_size, 3), 
                              include_top = False, 
                              weights = "imagenet")

    for layer in Xception_Model.layers:
      layer.trainable = False

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    x = MLP_classificador(Xception_Model.output, Units, Activation)

    Xception_model = Model(Xception_Model.input, x)

    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    Xception_model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ["accuracy"]
    )

    return Xception_model, Model_name, Model_name_letters

# ? VGG 16

def VGG16_pretrained(X_size: int, Y_size: int, Num_classes: int):
    """
    VGG16 configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning VGG16 model.
        string: Returning VGG16 Name.
    """

    Model_name = 'VGG16_Model'
    Model_name_letters = 'VGG16'

    VGG16_Model = VGG16(input_shape = (X_size, Y_size, 3), 
                        include_top = False, 
                        weights = "imagenet")

    for layer in VGG16_Model.layers:
      layer.trainable = False

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    x = MLP_classificador(VGG16_Model.output, Units, Activation)

    VGG16_model = Model(VGG16_Model.input, x)

    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    VGG16_model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ["accuracy"]
    )

    return VGG16_model, Model_name, Model_name_letters

# ? VGG 19

def VGG19_pretrained(X_size: int, Y_size: int, Num_classes: int):
    """
    VGG19 configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning VGG19 model.
        string: Returning VGG19 Name.
    """

    Model_name = 'VGG19_Model'
    Model_name_letters = 'VGG19'

    VGG19_Model = VGG19(input_shape = (X_size, Y_size, 3), 
                        include_top = False, 
                        weights = "imagenet")

    for layer in VGG19_Model.layers:
      layer.trainable = False

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    x = MLP_classificador(VGG19_Model.output, Units, Activation)
    
    VGG19_Model = Model(VGG19_Model.input, x)

    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    VGG19_Model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ["accuracy"]
    )

    return VGG19_Model, Model_name, Model_name_letters

# ? InceptionV3

def InceptionV3_pretrained(X_size: int, Y_size: int, Num_classes: int):
    """
    InceptionV3 configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning InceptionV3 model.
        string: Returning InceptionV3 Name.
    """

    Model_name = 'InceptionV3_Model'
    Model_name_letters = 'IV3'

    InceptionV3_Model = InceptionV3(input_shape = (X_size, Y_size, 3), 
                                    include_top = False, 
                                    weights = "imagenet")

    for layer in InceptionV3_Model.layers:
      layer.trainable = False

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    x = MLP_classificador(InceptionV3_Model.output, Units, Activation)

    InceptionV3_model = Model(InceptionV3_Model.input, x)

    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    InceptionV3_model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ["accuracy"]
    )

    return InceptionV3_model, Model_name, Model_name_letters

# ? DenseNet121

def DenseNet121_pretrained(X_size: int, Y_size: int, Num_classes: int):
    """
    DenseNet121 configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning DenseNet121 model.
        string: Returning DenseNet121 Name.
    """

    Model_name = 'DenseNet121_Model'
    Model_name_letters = 'DN121'

    DenseNet121_Model = DenseNet121(input_shape = (X_size, Y_size, 3), 
                                    include_top = False, 
                                    weights = "imagenet")

    for layer in DenseNet121_Model.layers:
      layer.trainable = False

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    x = MLP_classificador(DenseNet121_Model.output, Units, Activation)

    DenseNet121_model = Model(DenseNet121_Model.input, x)

    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    DenseNet121_model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = Loss,
        metrics = ["accuracy"]
    )

    return DenseNet121_model, Model_name, Model_name_letters

# ? DenseNet201

def DenseNet201_pretrained(X_size: int, Y_size: int, Num_classes: int):
    """
    DenseNet201 configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning DenseNet201 model.
        string: Returning DenseNet201 Name.
    """

    Model_name = 'DenseNet201_Model'
    Model_name_letters = 'DN201'

    DenseNet201_Model = DenseNet201(input_shape = (X_size, Y_size, 3), 
                                    include_top = False, 
                                    weights = "imagenet")

    for layer in DenseNet201_Model.layers:
      layer.trainable = False

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    x = MLP_classificador(DenseNet201_Model.output, Units, Activation)

    DenseNet201_model = Model(DenseNet201_Model.input, x)

    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    DenseNet201_model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ["accuracy"]
    )

    return DenseNet201_model, Model_name, Model_name_letters

# ? Custom AlexNet12

def CustomCNNAlexNet12_Model(X_size: int, Y_size: int, Num_classes: int):
    """
    CustomCNNAlexNet12 configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning CustomCNNAlexNet12 model.
        string: Returning CustomCNNAlexNet12 Name.
    """

    Model_name = 'CustomAlexNet12_Model'
    Model_name_letters = 'CAN12'

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    CustomCNN_Model = Input(shape = (X_size, Y_size, 3))

    x = CustomCNN_Model
   
    x = Conv2D(filters = 96, kernel_size = (11, 11), strides = (4,4), activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(x)   
    
    x = Conv2D(filters = 256, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)

    x = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3), strides=(2, 2))(x)

    x = Flatten()(x) 

    x = Dense(4096, activation = 'relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(4096, activation = 'relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(Units, activation = Activation)(x)

    CustomLeNet5_model = Model(CustomCNN_Model, x)
    
    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    CustomLeNet5_model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ["accuracy"]
    )

    return CustomLeNet5_model, Model_name, Model_name_letters

# ? Custom AlexNet12 Tunner

def CustomCNNAlexNet12Tunner_Model(X_size: int, Y_size: int, Num_classes: int, hp):
    """
    CustomCNNAlexNet12 configuration.

    Args:
        X_size (int): X's size value.
        Y_size (int): Y's size value.
        Num_classes (int): Number total of classes.

    Returns:
        _type_: _description_
        string: Returning CustomCNNAlexNet12 model.
        string: Returning CustomCNNAlexNet12 Name.
    """

    Model_name = 'CustomAlexNet12_Model'
    Model_name_letters = 'CAN12'

    if Num_classes == 2:
      Activation = 'sigmoid'
      Loss = "binary_crossentropy"
      Units = 1
    else:
      Activation = 'softmax'
      Units = Num_classes
      Loss = "categorical_crossentropy"
      #loss = "KLDivergence"

    CustomCNN_Model = Input(shape = (X_size, Y_size, 3))

    x = CustomCNN_Model
   
    x = Conv2D(filters = 96, kernel_size = (11, 11), strides = (4,4), activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(x)   
    
    x = Conv2D(filters = 256, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)

    x = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3), strides=(2, 2))(x)

    x = Flatten()(x) 

    x = Dense(hp.Choice('units', [32, 64, 256, 512, 1024, 2048, 4096]), activation = 'relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(hp.Choice('units', [32, 64, 256, 512, 1024, 2048, 4096]), activation = 'relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(Units, activation = Activation)(x)

    CustomLeNet5_model = Model(CustomCNN_Model, x)
    
    Opt = Adam(learning_rate = 0.0001, beta_1 = 0.5)

    CustomLeNet5_model.compile(
        optimizer = Opt,
        loss = Loss,
        metrics = ["accuracy"]
    )

    return CustomLeNet5_model, Model_name, Model_name_letters