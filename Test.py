import os
import numpy as np

from Final_Code_3_Data_Augmentation import DataAugmentation

def preprocessing_DataAugmentation_CNN(Folder_paths, Folder_destination, Labels_path, Iters):
    
    # * 
    Data_agumentation = []
    Image_class_value = []

    
    print(len(Folder_paths))
    print(len(Labels_path))
    print(len(Iters))

    # * List to add images and labels.
    Images = [None] * len(Folder_paths)
    Labels = [None] * len(Folder_paths)

    # * General parameters
    #Iter_normal = 20 
    #Iter_Clasification = 40 

    #Iter_normal = 18 
    #Iter_Clasification = 34

    #Iter_one = 18 
    #Iter_two = 34  

    #Images_class = 0 
    #Images_class = 1 

    for i in range(len(Folder_paths)):
        print(i)
        Data_agumentation.append(DataAugmentation(Folder = Folder_paths[i], NewFolder = Folder_destination, Severity = Labels_path[i], Sampling = Iters[i], Label = i, Saveimages = True))
        Images[i], Labels[i] = Data_agumentation[i].data_augmentation()

    # * With this class we use the technique called data augmentation to create new images with their transformations
    #Data_augmentation_one = DataAugmentation(Folder = Folder_path_one, NewFolder = Folder_destination, Severity = Label_one, Sampling = Iter_one, Label = Images_class_one, Saveimages = True)
    #Data_augmentation_two = DataAugmentation(Folder = Folder_path_two, NewFolder = Folder_destination, Severity = Label_two, Sampling = Iter_two, Label = Images_class_two, Saveimages = True)

    #Images_one, Labels_one = Data_augmentation_one.data_augmentation()
    #Images_two, Labels_two = Data_augmentation_two.data_augmentation()

    # * Add the value in the lists already created

    #ALL_labels = [df for df in Labels[i]]

    #for i in range(len(Data_agumentation)):

        #Images_total = Images_total + Images[i]
        #Labels_total = np.concatenate((ALL_labels), axis = None)

    #print(len(Images_mass))
    #print(len(Images_calcification))

    for i in range(len(Data_agumentation)):
        #print(Images[i])
        print(Labels[i])
    #return Images_total, Labels_total

preprocessing_DataAugmentation_CNN(["D:\Mini-MIAS\CroppedBenignImages", "D:\Mini-MIAS\CroppedNormalImages"], "D:\Mini-MIAS\Test", ['Tumor', 'Normal'], [2, 2])
