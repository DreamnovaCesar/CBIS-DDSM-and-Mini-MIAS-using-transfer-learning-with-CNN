
from Final_Code_0_Libraries import *

from Final_Code_3_Data_Augmentation import DataAugmentation

# ? Data augmentation for CNN using RAM

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

def preprocessing_DataAugmentation_Biclass_CNN(Folder_path_one, Folder_path_two, Folder_destination, Label_one, Label_two):

    # * List to add images and labels.
    #Images = []
    #Labels = []

    # * General parameters
    #Iter_normal = 20 
    #Iter_Clasification = 40 

    #Iter_normal = 18 
    #Iter_Clasification = 34

    Iter_one = 18 
    Iter_two = 34  

    Images_class_one = 0 
    Images_class_two = 1 

    # * With this class we use the technique called data augmentation to create new images with their transformations
    Data_augmentation_one = DataAugmentation(Folder = Folder_path_one, NewFolder = Folder_destination, Severity = Label_one, Sampling = Iter_one, Label = Images_class_one, Saveimages = True)
    Data_augmentation_two = DataAugmentation(Folder = Folder_path_two, NewFolder = Folder_destination, Severity = Label_two, Sampling = Iter_two, Label = Images_class_two, Saveimages = True)

    Images_one, Labels_one = Data_augmentation_one.data_augmentation_test_images()
    Images_two, Labels_two = Data_augmentation_two.data_augmentation_test_images()

    # * Add the value in the lists already created

    Images_total = Images_one + Images_two
    Labels_total = np.concatenate((Labels_one, Labels_two), axis = None)

    print(Images_one)
    print(Images_two)

    #print(len(Images_mass))
    #print(len(Images_calcification))

    return Images_total, Labels_total

def preprocessing_DataAugmentation_Multiclass_CNN(Folder_path_one, Folder_path_two, Folder_path_three, Folder_destination, Label_one, Label_two, Label_three):

    # * List to add images and labels.
    #Images = []
    #Labels = []

    # * General parameters
    #Iter_normal = 2
    #Iter_benign = 70
    #Iter_calcification = 90 

    Iter_one = 25
    Iter_two = 4
    Iter_three = 3 

    #Iter_normal = 2
    #Iter_mass = 8
    #Iter_calcification = 10 

    Images_class_one = 0 
    Images_class_two = 1 
    Images_class_three = 2 

    # * With this class we use the technique called data augmentation to create new images with their transformations
    Data_augmentation_one = DataAugmentation(Folder = Folder_path_one, NewFolder = Folder_destination, Severity = Label_one, Sampling = Iter_one, Label = Images_class_one, Saveimages = True)
    Data_augmentation_two = DataAugmentation(Folder = Folder_path_two, NewFolder = Folder_destination, Severity = Label_two, Sampling = Iter_two, Label = Images_class_two, Saveimages = True)
    Data_augmentation_three = DataAugmentation(Folder = Folder_path_three, NewFolder = Folder_destination, Severity = Label_three, Sampling = Iter_two, Label = Images_class_three, Saveimages = True)

    Images_one, Labels_one = Data_augmentation_one.data_augmentation_test_images()
    Images_two, Labels_two = Data_augmentation_two.data_augmentation_test_images()
    Images_three, Labels_three = Data_augmentation_three.data_augmentation_test_images()

    # * Add the value in the lists already created

    Images_total = Images_one + Images_two + Images_three
    Labels_total = np.concatenate((Labels_one, Labels_two, Labels_three), axis = None)
    
    print(Images_one)
    print(Images_two)
    print(Images_three)

    #print(len(Images_normal))
    #print(len(Images_mass))
    #print(len(Images_calcification))

    return Images_total, Labels_total

# ? Data augmentation from folder, Splitting data required.

def preprocessing_DataAugmentation_Biclass_Folder(Folder_path, First_label, Second_label, First_number_iter, Second_number_iter):

    # * List to add images and labels.
    #Images = []
    #Labels = []

    # * General parameters
    #First_number_iter = 20 
    #Second_number_iter = 40 

    #First_number_iter = 18 #*
    #Second_number_iter = 34  #* 

    Total_files: int = 0
    Total_dir: int = 0

    #First_number_iter = 1 
    #Second_number_iter = 1  

    First_images_class: int = 0 
    Second_images_class: int = 1 

    Dir_total_training = []
    Dir_total_val = []
    Dir_total_test = []

    Folder_path_train_classes = []
    Folder_path_val_classes = []
    Folder_path_test_classes = []

    Folder_path_train ='{}/train/'.format(Folder_path)
    Folder_path_val ='{}/val/'.format(Folder_path)
    Folder_path_test ='{}/test/'.format(Folder_path)

    for Base, Dirs, Files in os.walk(Folder_path_train):
        print('Searching in : ', Base)
        for Dir in Dirs:
            Dir_total_training.append(Dir)
            Total_dir += 1
        for Index, File in enumerate(Files):
            Total_files += 1
    """

    for base, dirs, files in os.walk(Folder_path_test):
        print('Searching in : ', base)
        for dir in dirs:
            Dir_total_test.append(dir)
            Total_dir += 1
        for file in files:
            Total_files += 1

    
    for base, dirs, files in os.walk(Folder_path_val):
        print('Searching in : ', base)
        for dir in dirs:
            Dir_total_val.append(dir)
            Total_dir += 1
        for file in files:
            Total_files += 1
    """

    #print(Dir_total[0])
    #print(Dir_total[1])
    #print(len(Dir_total))
    #print(Total_dir)
    #print(Total_files)

    for Index, dir in enumerate(Dir_total_training):
        print(Index)
        Folder_path_train_classes.append('{}{}'.format(Folder_path_train, dir))
        print(Folder_path_train_classes[Index])
    """
    for Index, dir in enumerate(Dir_total_test):
        print(Index)
        Folder_path_test_classes.append('{}{}'.format(Folder_path_test, dir))
        print(Folder_path_test_classes[Index])

    
    for Index, dir in enumerate(Dir_total_val):
        print(Index)
        Folder_path_val_classes.append('{}{}'.format(Folder_path_val, dir))
        print(Folder_path_val_classes[Index])
    """
    # * With this class we use the technique called data augmentation to create new images with their transformations

    Data_augmentation_normal = DataAugmentation(Folder = Folder_path_train_classes[0], NewFolder = Folder_path_train_classes[0], Severity = First_label, Sampling = First_number_iter, Label = First_images_class, Saveimages = True)
    Data_augmentation_tumor = DataAugmentation(Folder = Folder_path_train_classes[1], NewFolder = Folder_path_train_classes[1], Severity = Second_label, Sampling = Second_number_iter, Label = Second_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation_same_folder()
    Images_Tumor, Labels_Tumor = Data_augmentation_tumor.data_augmentation_same_folder()

    # * Add the value in the lists already created

    Images_total = Images_Normal + Images_Tumor
    Labels_total = np.concatenate((Labels_Normal, Labels_Tumor), axis = None)

    print(len(Images_Normal))
    print(len(Images_Tumor))

    #print(len(Images_total))
    #print(len(Labels_total))

    """
    Data_augmentation_normal = DataAugmentation(Folder = Folder_path_test_classes[0], NewFolder = Folder_path_test_classes[0], Severity = Label_normal, Sampling = Iter_normal, Label = Normal_images_class, Saveimages = True)
    Data_augmentation_tumor = DataAugmentation(Folder = Folder_path_test_classes[1], NewFolder = Folder_path_test_classes[1], Severity = Label_tumor, Sampling = Iter_tumor, Label = Tumor_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation_same_folder()
    Images_Tumor, Labels_Tumor = Data_augmentation_tumor.data_augmentation_same_folder()

    # * Add the value in the lists already created

    Images_total = Images_Normal + Images_Tumor
    Labels_total = np.concatenate((Labels_Normal, Labels_Tumor), axis = None)

    print(len(Images_Normal))
    print(len(Images_Tumor))


    Data_augmentation_normal = DataAugmentation(Folder = Folder_path_val_classes[0], NewFolder = Folder_path_val_classes[0], Severity = First_label, Sampling = First_number_iter, Label = First_images_class, Saveimages = True)
    Data_augmentation_tumor = DataAugmentation(Folder = Folder_path_val_classes[1], NewFolder = Folder_path_val_classes[1], Severity = Second_label, Sampling = Second_number_iter, Label = Second_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation_test_images()
    Images_Tumor, Labels_Tumor = Data_augmentation_tumor.data_augmentation_test_images()

    # * Add the value in the lists already created

    Images_total = Images_Normal + Images_Tumor
    Labels_total = np.concatenate((Labels_Normal, Labels_Tumor), axis = None)

    print(len(Images_total))
    print(len(Labels_total))
    """

def preprocessing_DataAugmentation_Multiclass_Folder(Folder_path, First_label, Second_label, Third_label, First_number_iter, Second_number_iter, Third_number_iter):

    # * List to add images and labels.
    #Images = []
    #Labels = []

    # * General parameters

    Total_files:int = 0
    Total_dir:int = 0

    #First_number_iter = 10
    vSecond_number_iter = 1
    #Third_number_iter = 1 

    First_images_class:int = 0 
    Second_images_class:int = 1
    Third_images_class:int = 1 

    Dir_total_training = []
    Dir_total_test = []
    Dir_total_val = []

    Folder_path_train_classes = []
    Folder_path_test_classes = []
    Folder_path_val_classes = []

    Folder_path_train ='{}/train/'.format(Folder_path)
    Folder_path_test ='{}/test/'.format(Folder_path)
    Folder_path_val ='{}/val/'.format(Folder_path)

    for Base, Dirs, Files in os.walk(Folder_path_train):
        print('Searching in : ', Base)
        for Dir in Dirs:
            Dir_total_training.append(Dir)
            Total_dir += 1
        for Index, File in enumerate(Files):
            Total_files += 1
    """
    for base, dirs, files in os.walk(Folder_path_test):
        print('Searching in : ', base)
        for dir in dirs:
            Dir_total_test.append(dir)
            Total_dir += 1
        for file in files:
            Total_files += 1

    for base, dirs, files in os.walk(Folder_path_val):
        print('Searching in : ', base)
        for dir in dirs:
            Dir_total_val.append(dir)
            Total_dir += 1
        for file in files:
            Total_files += 1
    """
    #print(Dir_total[0])
    #print(Dir_total[1])
    #print(len(Dir_total))
    #print(Total_dir)
    #print(Total_files)

    for Index, dir in enumerate(Dir_total_training):
        print(Index)
        Folder_path_train_classes.append('{}{}'.format(Folder_path_train, dir))
        print(Folder_path_train_classes[Index])
    """
    for Index, dir in enumerate(Dir_total_test):
        print(Index)
        Folder_path_test_classes.append('{}{}'.format(Folder_path_test, dir))
        print(Folder_path_test_classes[Index])

    for Index, dir in enumerate(Dir_total_val):
        print(Index)
        Folder_path_val_classes.append('{}{}'.format(Folder_path_val, dir))
        print(Folder_path_val_classes[Index])
    """    
    # * With this class we use the technique called data augmentation to create new images with their transformations

    Data_augmentation_normal = DataAugmentation(Folder = Folder_path_train_classes[0], NewFolder = Folder_path_train_classes[0], Severity = First_label, Sampling = First_number_iter, Label = First_images_class, Saveimages = True)
    Data_augmentation_benign = DataAugmentation(Folder = Folder_path_train_classes[1], NewFolder = Folder_path_train_classes[1], Severity = Second_label, Sampling = Second_number_iter, Label = Second_images_class, Saveimages = True)
    Data_augmentation_malignant = DataAugmentation(Folder = Folder_path_train_classes[2], NewFolder = Folder_path_train_classes[2], Severity = Third_label, Sampling = Third_number_iter, Label = Third_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation_same_folder()
    Images_Benign, Labels_Benign = Data_augmentation_benign.data_augmentation_same_folder()
    Images_Malignant, Labels_Malignant = Data_augmentation_malignant.data_augmentation_same_folder()
    # * Add the value in the lists already created

    Images_total = Images_Normal + Images_Benign + Images_Malignant
    Labels_total = np.concatenate((Labels_Normal, Labels_Benign, Labels_Malignant), axis = None)
    
    #print(Images_Normal)
    #print(Images_Benign)
    #print(Images_Malignant)
    
    print(len(Images_Normal))
    print(len(Images_Benign))
    print(len(Images_Malignant))

    """
    Data_augmentation_normal = DataAugmentation(Folder = Folder_path_val_classes[0], NewFolder = Folder_path_val_classes[0], Severity = Label_benign, Sampling = Iter_benign, Label = Benign_images_class, Saveimages = True)
    Data_augmentation_benign = DataAugmentation(Folder = Folder_path_val_classes[1], NewFolder = Folder_path_val_classes[1], Severity = Label_malignant, Sampling = Iter_malignant, Label = Malignant_images_class, Saveimages = True)
    Data_augmentation_malignant = DataAugmentation(Folder = Folder_path_val_classes[2], NewFolder = Folder_path_val_classes[2], Severity = Label_normal, Sampling = Iter_normal, Label = Normal_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation_same_folder()
    Images_Benign, Labels_Benign = Data_augmentation_benign.data_augmentation_same_folder()
    Images_Malignant, Labels_Malignant = Data_augmentation_malignant.data_augmentation_same_folder()

    # * Add the value in the lists already created

    Images_total = Images_Normal + Images_Benign + Images_Malignant
    Labels_total = np.concatenate((Labels_Normal, Labels_Benign, Labels_Malignant), axis = None)
    
    print(len(Images_Normal))
    print(len(Images_Benign))
    print(len(Images_Malignant))
    """
