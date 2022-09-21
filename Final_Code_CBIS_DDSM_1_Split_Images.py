

from Final_Code_1_General_Functions import CBIS_DDSM_CSV_severity_labeled
from Final_Code_1_General_Functions import CBIS_DDSM_split_data

def preprocessing_SplitCalcMass():

    Severity = 9

    Calcification = 'Calficication'
    Mass = 'Mass'

    Test = 'Test'
    Training = 'Training'

    Calc_Description_Test = "D:\CBIS-DDSM\CBIS-DDSM Final\CBIS-DDSM_Calcification\CBIS-DDSM_Calcification_Test\calc_case_description_test_set.csv"
    Calc_Description_Training = "D:\CBIS-DDSM\CBIS-DDSM Final\CBIS-DDSM_Calcification\CBIS-DDSM_Calcification_Training\calc_case_description_train_set.csv"

    Mass_Description_Test = "D:\CBIS-DDSM\CBIS-DDSM Final\CBIS-DDSM_Mass\CBIS-DDSM_Mass_Test\mass_case_description_test_set.csv"
    Mass_Description_Training = "D:\CBIS-DDSM\CBIS-DDSM Final\CBIS-DDSM_Mass\CBIS-DDSM_Mass_Training\mass_case_description_train_set.csv"

    Dataframe_Calc_Severity_Test = CBIS_DDSM_CSV_severity_labeled(Calc_Description_Test, Severity, 1)
    Dataframe_Calc_Severity_Training = CBIS_DDSM_CSV_severity_labeled(Calc_Description_Training, Severity, 1)

    Dataframe_Mass_Severity_Test = CBIS_DDSM_CSV_severity_labeled(Mass_Description_Test, Severity, 2)
    Dataframe_Mass_Severity_Training = CBIS_DDSM_CSV_severity_labeled(Mass_Description_Training, Severity, 2)

    #Folder_CSV, Folder, Folder_total_benign, Folder_benign, Folder_benign_wc, Folder_malignant, Folder_abnormal, Dataframe, Severity, Phase
    Dataframe = CBIS_DDSM_split_data(   General_Data_CSV, Calc_Test_Patches_Resize, Calc_NT_All_Benign_Images, Calc_NT_Benign_Images, Calc_NT_Benign_WC_Images, Calc_NT_Malignant_Images, Calc_NT_Abnormal_Images, 
                                        Dataframe_Calc_Severity_Test, Calcification, Test)
    
    Dataframe = CBIS_DDSM_split_data(   General_Data_CSV, Calc_Training_Patches_Resize, Calc_NT_All_Benign_Images, Calc_NT_Benign_Images, Calc_NT_Benign_WC_Images, Calc_NT_Malignant_Images, Calc_NT_Abnormal_Images, 
                                        Dataframe_Calc_Severity_Training, Calcification, Training)

    Dataframe = CBIS_DDSM_split_data(   General_Data_CSV, Mass_Test_Patches_Resize, Mass_NT_All_Benign_Images, Mass_NT_Benign_Images, Mass_NT_Benign_WC_Images, Mass_NT_Malignant_Images, Mass_NT_Abnormal_Images, 
                                        Dataframe_Mass_Severity_Test, Mass, Test)
    
    Dataframe = CBIS_DDSM_split_data(   General_Data_CSV, Mass_Training_Patches_Resize, Mass_NT_All_Benign_Images, Mass_NT_Benign_Images, Mass_NT_Benign_WC_Images, Mass_NT_Malignant_Images, Mass_NT_Abnormal_Images, 
                                        Dataframe_Mass_Severity_Training, Mass, Training)


    Dataframe = CBIS_DDSM_split_data(   General_Data_CSV, Calc_Test_Patches_Resize_NO, Calc_NO_All_Benign_Images, Calc_NO_Benign_Images, Calc_NO_Benign_WC_Images, Calc_NO_Malignant_Images, Calc_NO_Abnormal_Images, 
                                        Dataframe_Calc_Severity_Test, Calcification, Test)
    
    Dataframe = CBIS_DDSM_split_data(   General_Data_CSV, Calc_Training_Patches_Resize_NO, Calc_NO_All_Benign_Images, Calc_NO_Benign_Images, Calc_NO_Benign_WC_Images, Calc_NO_Malignant_Images, Calc_NO_Abnormal_Images, 
                                        Dataframe_Calc_Severity_Training, Calcification, Training)

    Dataframe = CBIS_DDSM_split_data(   General_Data_CSV, Mass_Test_Patches_Resize_NO, Mass_NO_All_Benign_Images, Mass_NO_Benign_Images, Mass_NO_Benign_WC_Images, Mass_NO_Malignant_Images, Mass_NO_Abnormal_Images, 
                                        Dataframe_Mass_Severity_Test, Mass, Test)
    
    Dataframe = CBIS_DDSM_split_data(   General_Data_CSV, Mass_Training_Patches_Resize_NO, Mass_NO_All_Benign_Images, Mass_NO_Benign_Images, Mass_NO_Benign_WC_Images, Mass_NO_Malignant_Images, Mass_NO_Abnormal_Images, 
                                        Dataframe_Mass_Severity_Training, Mass, Training)

    ########## ########## ########## ########## ########## ########## ########## ########## ########## ##########