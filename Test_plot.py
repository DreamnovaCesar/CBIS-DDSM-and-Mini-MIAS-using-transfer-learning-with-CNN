from Final_Code_1_General_Functions import *


ROC = "D:\Test\Biclass_Folder_Data_Models_TEST\Biclass_Folder_Data_Model_MN_TEST\Biclass_Dataframe_ROC_Curve_Values_MobileNet_Model_TEST.csv"
H = "D:\Test\Biclass_Folder_Data_Models_TEST\Biclass_Folder_Data_Model_MN_TEST\Biclass_Logger_MobileNet_Model_TEST.csv"
CM = "D:\Test\Biclass_Folder_Data_Models_TEST\Biclass_Folder_Data_Model_MN_TEST\Biclass_Dataframe_Confusion_Matrix_MobileNet_Model_TEST.csv"

Fol = "D:\Test\Biclass_Folder_Data_Models_TEST\Biclass_Folder_Data_Model_MobileNetV3Small_Model_TEST"

f = FigurePlot(folder = Fol, title = 'MobileNetV3', SI = False, SF = True, height = 12, width = 12, annot_kws = 20, font = 1, CMdf = CM, Hdf = H, ROCdf = ROC)

df = f.Roc_curve_dataframe_property

print(df)


