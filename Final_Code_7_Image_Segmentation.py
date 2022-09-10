import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

from Mini_MIAS_2_General_Functions import sort_images
from Mini_MIAS_2_General_Functions import remove_all_files

"""
img = cv2.imread("D:\CBIS-DDSM\CBIS-DDSM Final\CBIS_DDSM_NO_Images_Biclass\Mass_NO_AbnormalImages_DA\Mass-Training_P_00018_RIGHT_MLO_1_Benign_0_Rotation_Augmentation.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8,8))
plt.imshow(img, cmap="gray")
plt.axis('off')
plt.title("Original Image")
plt.show()

ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
plt.figure(figsize=(8,8))
plt.imshow(thresh, cmap="gray")
plt.axis('off')
plt.title("Threshold Image")
plt.show()


kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations = 15)
bg = cv2.dilate(closing, kernel, iterations = 1)
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
ret, fg = cv2.threshold(dist_transform, 0.02*dist_transform.max(), 255, 0)
cv2.imshow('image', fg)
plt.figure(figsize = (8,8))
plt.imshow(fg,cmap = "gray")
plt.axis('off')
plt.title("Segmented Image")
plt.show()
"""

def image_segmentation(Folder, NewFolder, Label):
    
    # * Remove all the files in the new folder using this function
    remove_all_files(NewFolder)

    # * Lists to save the values of the labels and the filename and later use them for a dataframe
    #Images = [] 
    Labels = []
    All_filenames = []

    os.chdir(Folder)

    # * Using sort function
    Sorted_files, Total_images = sort_images(Folder)
    Count = 1

    # * Reading the files
    for File in Sorted_files:

      # * Extract the file name and format
      Filename, Format  = os.path.splitext(File)

      # * Read extension files
      if File.endswith(Format): 
        
        print(f"Working with {Count} of {Total_images} images ✅")
        print(f"Working with {Filename} ✅")

        # * Resize with the given values
        Path_file = os.path.join(Folder, File)
        Image = cv2.imread(Path_file)
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(Image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations = 15)
        bg = cv2.dilate(closing, kernel, iterations = 1)
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
        ret, fg = cv2.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0)

        # * Name the new file
        Filename_and_technique = Filename + '_segmentation'
        New_name_filename = Filename_and_technique + Format
        New_folder = os.path.join(NewFolder, New_name_filename)
        
        # * Save the image in a new folder
        cv2.imwrite(New_folder, fg)
        
        # * Save the values of labels and each filenames
        #Images.append(Normalization_Imagen)
        Labels.append(Label)
        All_filenames.append(Filename_and_technique)

        Count += 1
    
    # * Return the new dataframe with the new data
    Dataframe = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'Labels':Labels})

    return Dataframe


training = r"D:\CBIS-DDSM\CBIS-DDSM Final\CBIS_DDSM_NO_Images_Biclass_Split\train\Calc_NO_AbnormalImages_DA"
training_seg = r"D:\CBIS-DDSM\CBIS-DDSM Final\Training"

test = r"D:\CBIS-DDSM\CBIS-DDSM Final\CBIS_DDSM_NO_Images_Biclass_Split\test\Calc_NO_AbnormalImages_DA"
test_seg = r"D:\CBIS-DDSM\CBIS-DDSM Final\Test"

image_segmentation(training, training_seg, 0)
image_segmentation(test, test_seg, 1)