
import numpy as np

from Final_Code_Preprocessing_1_CNN_Models import Testing_CNN_Models_Biclass_From_Folder

#Model_CNN = (Model_pretrained, Model_pretrained)
Model_CNN = (13, 14)

def main():
    Testing_CNN_Models_Biclass_From_Folder(Model_CNN, 'D:\Mini-MIAS\Mini_MIAS_NO_Cropped_Images_Biclass' + '_Split', 'TEST')

if __name__ == "__main__":
    main()