from Final_Code_1_General_Functions import *

"D:\Test\Biclass_Folder_Data_Models_TEST\Biclass_Folder_Data_Model_MobileNetV3Small_Model_TEST\Biclass_Dataframe_CNN_Folder_Data_TEST.csv"
ROC = "D:\Test\Biclass_Folder_Data_Models_TEST\Biclass_Folder_Data_Model_MobileNetV3Small_Model_TEST\Biclass_Dataframe_ROC_Curve_Values_MNV3S_TEST.csv"
"D:\Test\Biclass_Folder_Data_Models_TEST\Biclass_Folder_Data_Model_MobileNetV3Small_Model_TEST\Biclass_MNV3S_TEST_Best_Model_Weights.h5"
H = "D:\Test\Biclass_Folder_Data_Models_TEST\Biclass_Folder_Data_Model_MobileNetV3Small_Model_TEST\Biclass_MNV3S_TEST_logger.csv"
CM = "D:\Test\Biclass_Folder_Data_Models_TEST\Biclass_Folder_Data_Model_MobileNetV3Small_Model_TEST\Dataframe_Confusion_Matrix_Biclass_MNV3S_TEST.csv"

Fol = "D:\Test\Biclass_Folder_Data_Models_TEST\Biclass_Folder_Data_Model_MobileNetV3Small_Model_TEST"

f = FigurePlot(folder = Fol, title = 'MobileNetV3', SI = False, SF = True, height = 12, width = 12, annot_kws = 20, font = 1, CMdf = CM, Hdf = H, ROCdf = ROC)

f.figure_plot_four()
f.figure_plot_CM()
