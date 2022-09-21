
from Final_Code_1_General_Functions import DCM_format

def preprocessing_ChangeFormat():

    Calcification = 'Calficication'
    Mass = 'Mass'

    Test = 'Test'
    Training = 'Training'


    # * With this class we change the format of each image for a new one
    DCM_Calc_Test = DCM_format(Folder = Calc_Test, Datafolder = Calc_CBIS_DDSM_Test, Normalfolder = Calc_Test_Patches, 
                                Resizefolder = Calc_Test_Patches_Resize, Normalizefolder = Calc_Test_Patches_Resize_NO, 
                                    Severity = Calcification, Phase = Test)

    DCM_Calc_Training = DCM_format(Folder = Calc_Training, Datafolder = Calc_CBIS_DDSM_Training, Normalfolder = Calc_Training_Patches, 
                                    Resizefolder = Calc_Training_Patches_Resize, Normalizefolder = Calc_Training_Patches_Resize_NO, 
                                        Severity = Calcification, Phase = Training)

    DCM_Mass_Test = DCM_format(Folder = Mass_Test, Datafolder = Mass_CBIS_DDSM_Test, Normalfolder = Mass_Test_Patches, 
                                Resizefolder = Mass_Test_Patches_Resize, Normalizefolder = Mass_Test_Patches_Resize_NO,    
                                    Severity = Mass, Phase = Test)

    DCM_Mass_Training = DCM_format(Folder = Mass_Training, Datafolder = Mass_CBIS_DDSM_Training, Normalfolder = Mass_Training_Patches, 
                            Resizefolder = Mass_Training_Patches_Resize, Normalizefolder = Mass_Training_Patches_Resize_NO, 
                                Severity = Mass, Phase = Training)


    DCM_Calc_Test.DCM_change_format()
    DCM_Calc_Training.DCM_change_format()
    DCM_Mass_Test.DCM_change_format()
    DCM_Mass_Training.DCM_change_format()