from ast import Return
from Final_Code_0_Libraries import *

from functools import wraps

################################################## ? Class decorators

# ? Sort Files
def sort_images(Folder_path: str) -> tuple[list[str], int]: 
    """
    Sort the filenames of the obtained folder path.

    Args:
        Folder_path (str): Folder path obtained.

    Returns:
        list[str]: Return all files sorted.
        int: Return the number of images inside the folder.
    """
    
    # * Folder attribute (ValueError, TypeError)
    if Folder_path == None:
        raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_path, str):
        raise TypeError("Folder must be a string") #! Alert

    Asterisks:int = 60
    # * This function sort the files and show them

    Number_images: int = len(os.listdir(Folder_path))
    print("\n")
    print("*" * Asterisks)
    print('Images: {}'.format(Number_images))
    print("*" * Asterisks)
    Files: list[str] = os.listdir(Folder_path)
    print("\n")

    Sorted_files: list[str] = sorted(Files)

    for Index, Sort_file in enumerate(Sorted_files):
        print('Index: {} ---------- {} ✅'.format(Index, Sort_file))

    print("\n")

    return Sorted_files, Number_images

#################################################################################################### ? Class 

# ?

class Utilities(object):

    # ? Get the execution time of each function
    @staticmethod  
    def timer_func(func):  
        @wraps(func)  
        def wrapper(self, *args, **kwargs):  

            # * Obtain the executed time of the function

            Asterisk = 60;

            t1 = time.time();
            result = func(self, *args, **kwargs);
            t2 = time.time();

            print("\n");
            print("*" * Asterisk);
            print('Function {} executed in {:.4f}'.format(func.__name__, t2 - t1));
            print("*" * Asterisk);
            print("\n");

            return result
        return wrapper
    
    # ? Detect fi GPU exist in your PC for CNN Decorator
    @staticmethod  
    def detect_GPU(func):  
        @wraps(func)  
        def wrapper(self, *args, **kwargs):  

            # * Obtain the executed time of the function
            GPU_name = tf.test.gpu_device_name();
            GPU_available = tf.config.list_physical_devices();
            print("\n");
            print(GPU_available);
            print("\n");
            #if GPU_available == True:
                #print("GPU device is available")
            if "GPU" not in GPU_name:
                print("GPU device not found");
                print("\n");
            print('Found GPU at: {}'.format(GPU_name));
            print("\n");

            result = func(self, *args, **kwargs);

            return result
        return wrapper

# ? Random remove all files in folder

class RemoveFiles(Utilities):

    # ? Remove all files inside a dir
    @staticmethod
    @Utilities.timer_func 
    def remove_all_files(func):  
        @wraps(func)  
        def wrapper(self, *args, **kwargs):  

            # * 
            for File in os.listdir(self._Folder_path):
                Filename, Format  = os.path.splitext(File);
                print('Removing: {} . {} ✅'.format(Filename, Format));
                os.remove(os.path.join(self._Folder_path, File));

            result = func(self, *args, **kwargs)

            return result
        return wrapper

    def __init__(self, **kwargs) -> None:

        # * Instance attributes (Protected)
        self._Folder_path = kwargs.get('folder', None);
        self._Number_Files_to_remove = kwargs.get('NFR', None);

    def __repr__(self):

        kwargs_info = "[{}, {}]".format(self._Folder_path, self._Number_Files_to_remove);

        return kwargs_info

    def __str__(self):
        pass
    
    # * Folder_path attribute
    @property
    def Folder_path_property(self):
        return self._Folder_path

    @Folder_path_property.setter
    def Folder_path_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Folder_path must be a string") #! Alert
        self._Folder_path = New_value;
    
    @Folder_path_property.deleter
    def Folder_path_property(self):
        print("Deleting Folder_path...");
        del self._Folder_path

    # * Files_to_remove attribute
    @property
    def Files_to_remove_property(self):
        return self._Files_to_remove

    @Files_to_remove_property.setter
    def Files_to_remove_property(self, New_value):
        if not isinstance(New_value, int):
            raise TypeError("Files_to_remove must be a integer") #! Alert
        self._Files_to_remove = New_value;
    
    @Files_to_remove_property.deleter
    def Files_to_remove_property(self):
        print("Deleting Files_to_remove...");
        del self._Files_to_remove

    # ? Remove all files inside a dir
    @Utilities.timer_func
    def remove_all_files(self) -> None:
        """
        Remove all files inside the folder path obtained.

        Args:
            Folder_path (str): Folder path obtained.

        Returns:
            None
        """
        
        # * Folder attribute (ValueError, TypeError)
        if self._Folder_path == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self._Folder_path, str):
            raise TypeError("Folder must be a string") #! Alert

        # * This function will remove all the files inside a folder
        for File in os.listdir(self._Folder_path):
            Filename, Format  = os.path.splitext(File);
            print('Removing: {} . {} ✅'.format(Filename, Format));
            os.remove(os.path.join(self._Folder_path, File));

    # ? Remove all files inside a dir
    @Utilities.timer_func
    def remove_random_files(self) -> None:
        """
        Remove all files inside the folder path obtained.

        Args:
            Folder_path (str): Folder path obtained.

        Returns:
            None
        """

        # * This function will remove all the files inside a folder
        Files = os.listdir(self._Folder_path);

            #Filename, Format = os.path.splitext(File)

        for File_sample in sample(Files, self._Number_Files_to_remove):
            print(File_sample);
            #print('Removing: {}{} ✅'.format(Filename, Format));
            os.remove(os.path.join(self._Folder_path, File_sample));

# ?

class Generator(object):

    def __init__(self, **kwargs) -> None:

        # * Instance attributes (Private)
        self.__Folder_path = kwargs.get('folder', None);
        self.__Folders_name = kwargs.get('FN', None);
        self.__Iteration = kwargs.get('iter', None);
        self.__Save_dataframe = kwargs.get('SD', None);

    # ? Create folders
    @Utilities.timer_func
    def create_folders(self) -> pd.DataFrame: 
        """
        _summary_

        _extended_summary_

        Args:
            Folder_path (str): _description_
            Folder_name (list[str]): _description_
            CSV_name (str): _description_
        """

        # *
        Path_names = [];
        Path_absolute_dir = [];

        # *
        if(len(self.__Folders_name) >= 2):
            
            # *
            for i, Path_name in enumerate(self.__Folders_name):

                # *
                Folder_path_new = r'{}\{}'.format(self.__Folder_path, self.__Folders_name[i]);
                print(Folder_path_new);

                Path_names.append(Path_name);
                Path_absolute_dir.append(Folder_path_new);

                Exist_dir = os.path.isdir(Folder_path_new) ;

            if Exist_dir == False:
                os.mkdir(Folder_path_new);
            else:
                print('Path: {} exists, use another name for it'.format(self.__Folders_name[i]));

        else:

            # *
            Folder_path_new = r'{}\{}'.format(self.__Folder_path, self.__Folders_name);
            print(Folder_path_new);

            Path_names.append(self.__Folders_name);
            Path_absolute_dir.append(Folder_path_new);

            Exist_dir = os.path.isdir(Folder_path_new) ;

            if Exist_dir == False:
                os.mkdir(Folder_path_new);
            else:
                print('Path: {} exists, use another name for it'.format(self.__Folders_name));

        # *
        if(self.__Save_dataframe == True):
            Dataframe = pd.DataFrame({'Names':Path_names, 'Path names':Path_absolute_dir});
            Dataframe_name = 'Dataframe_path_names.csv'.format();
            Dataframe_folder = os.path.join(self.__Folder_path, Dataframe_name);

            #Exist_dataframe = os.path.isfile(Dataframe_folder)

            Dataframe.to_csv(Dataframe_folder);

        return Dataframe

    # ? Create folders
    @Utilities.timer_func
    def creating_data_students(self) -> pd.DataFrame: 
        
        # * Tuples for random generation.
        Random_Name = ('Tom', 'Nick', 'Chris', 'Jack', 'Thompson');
        Random_Classroom = ('A', 'B', 'C', 'D', 'E');
        
        # * Column names to create the DataFrame
        Columns_names = ['Name', 'Age', 'Classroom', 'Height', 'Math', 'Chemistry', 'Physics', 'Literature'];

        # *
        Dataframe = pd.DataFrame(Columns_names)

        for i in range(self.__Iteration):

            # *
            New_row = { 'Name':random.choice(Random_Name),
                        'Age':randint(16, 26),
                        'Classroom':random.choice(Random_Classroom),
                        'Height':randint(160, 195),
                        'Math':randint(70, 100),
                        'Chemistry':randint(70, 100),
                        'Physics':randint(70, 100),
                        'Literature':randint(70, 100)};

            Dataframe = Dataframe.append(New_row, ignore_index = True);

            # *
            print('Iteration complete: {}'.format(i));

        # *
        if(self.__Save_dataframe == True):
            Dataframe_name = 'Dataframe_students_data.csv'.format();
            Dataframe_folder = os.path.join(self.__Folder_path, Dataframe_name);
            Dataframe.to_csv(Dataframe_folder);

        return Dataframe

# ? Generate keys

class SecurityFiles(Utilities):

    def __init__(self, **kwargs) -> None:
        
        # * Instance attributes (Private)
        self.__Folder_path = kwargs.get('folder', None);
        self.__Number_keys = kwargs.get('NK', 2);
        self.__Key_path = kwargs.get('KP', None);
        self.__Keys_path = kwargs.get('KsP', None);
        self.__Key_chosen = kwargs.get('KC', None);
        self.__Key_random = kwargs.get('KR', None);

    @Utilities.timer_func
    def generate_key(self) -> None: 
        
        # *
        Names = [];
        Keys = [];
        
        # * key generation
        for i in range(self.__Number_keys):

            Key = Fernet.generate_key()
            
            print('Key created: {}'.format(Key))

            Key_name = 'filekey_{}'.format(i)
            Key_path_name = '{}/filekey_{}.key'.format(self.__Folder_path, i)

            Keys.append(Key)
            Names.append(Key_name)

            with open(Key_path_name, 'wb') as Filekey:
                Filekey.write(Key)

            Dataframe_keys = pd.DataFrame({'Name':Names, 'Keys':Keys})

            Dataframe_Key_name = 'Dataframe_filekeys.csv'.format()
            Dataframe_Key_folder = os.path.join(self.__Folder_path, Dataframe_Key_name)

            Dataframe_keys.to_csv(Dataframe_Key_folder)

    # ? Encrypt files

    @Utilities.timer_func
    def encrypt_files(self) -> None:

        # * Folder attribute (ValueError, TypeError)
        if self.__Folder_path == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self.__Folder_path, str):
            raise TypeError("Folder must be a string") #! Alert

        Filenames = []

        if self.__Key_random == True:

            File = random.choice(os.listdir(self.__Keys_path))
            FilenameKey, Format = os.path.splitext(File)

            if File.endswith('.key'):

                try:
                    with open(self.__Keys_path + '/' + File, 'rb') as filekey:
                        Key = filekey.read()

                    Fernet_ = Fernet(Key)

                    #Key_name = 'MainKey.key'.format()
                    shutil.copy2(os.path.join(self.__Keys_path, File), self.__Key_chosen)

                    # * This function sort the files and show them

                    for Filename in os.listdir(self.__Folder_path):

                        Filenames.append(Filename)

                        with open(self.__Folder_path + '/' + Filename, 'rb') as File_: # open in readonly mode
                            Original_file = File_.read()
                        
                        Encrypted_File = Fernet_.encrypt(Original_file)

                        with open(self.__Folder_path + '/' + Filename, 'wb') as Encrypted_file:
                            Encrypted_file.write(Encrypted_File) 

                    with open(self.__Key_path_chosen + '/' + FilenameKey + '.txt', "w") as text_file:
                        text_file.write('The key {} open the next documents {}'.format(FilenameKey, Filenames))   

                except OSError:
                    print('Is not a key {} ❌'.format(str(File))) #! Alert

        elif self.__Key_random == False:

            Name_key = os.path.basename(self.__Key_path)
            Key_dir = os.path.dirname(self.__Key_path)

            if self.__Key_path.endswith('.key'):
                
                try: 
                    with open(self.__Key_path, 'rb') as filekey:
                        Key = filekey.read()

                    Fernet_ = Fernet(Key)

                    #Key_name = 'MainKey.key'.format()
                    shutil.copy2(os.path.join(Key_dir, Name_key), self.__Key_chosen)

                    # * This function sort the files and show them

                    for Filename in os.listdir(self.__Folder_path):

                        Filenames.append(Filename)

                        with open(self.__Folder_path + '/' + Filename, 'rb') as File: # open in readonly mode
                            Original_file = File.read()
                        
                        Encrypted_File = Fernet_.encrypt(Original_file)

                        with open(self.__Folder_path + '/' + Filename, 'wb') as Encrypted_file:
                            Encrypted_file.write(Encrypted_File)

                    with open(self.__Key_path_chosen + '/' + Name_key + '.txt', "w") as text_file:
                        text_file.write('The key {} open the next documents {}'.format(Name_key, Filenames))  

                except OSError:
                    print('Is not a key {} ❌'.format(str(self.__Key_path))) #! Alert

    # ? Decrypt files

    @Utilities.timer_func
    def decrypt_files(self) -> None: 


        # * Folder attribute (ValueError, TypeError)
        if self.__Folder_path == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self.__Folder_path, str):
            raise TypeError("Folder must be a string") #! Alert

        Key_dir = os.path.dirname(self.__Key_path)
        Key_file = os.path.basename(self.__Key_path)

        Filename_key, Format = os.path.splitext(Key_file)

        Datetime = datetime.datetime.now()

        with open(self.__Key_path, 'rb') as Filekey:
            Key = Filekey.read()

        Fernet_ = Fernet(Key)

        # * This function sort the files and show them

        if Filename_key.endswith('.key'):

            try:
                for Filename in os.listdir(self.__Folder_path):

                    print(Filename)

                    with open(self.__Folder_path + '/' + Filename, 'rb') as Encrypted_file: # open in readonly mode
                        Encrypted = Encrypted_file.read()
                    
                    Decrypted = Fernet_.decrypt(Encrypted)

                    with open(self.__Folder_path + '/' + Filename, 'wb') as Decrypted_file:
                        Decrypted_file.write(Decrypted)

                with open(Key_dir + '/' + Key_file + '.txt', "w") as text_file:
                        text_file.write('Key used. Datetime: {} '.format(Datetime))  

            except OSError:
                    print('Is not a key {} ❌'.format(str(self.__Key_path))) #! Alert


# ? ####################################################### Mini-MIAS #######################################################

# ? Extract the mean of each column

@Utilities.timer_func
def extract_mean_from_images(Dataframe:pd.DataFrame, Column:int) -> int:
  """
  Extract the mean from the values of the whole dataset using its dataframe.

  Args:
      Dataframe (pd.DataFrame): Dataframe with all the data needed(Mini-MIAS in this case).
      Column (int): The column number where it extracts the values.
  Returns:
      int: Return the mean from the column values.
  """

  # * This function will obtain the main of each column

  List_data_mean = []

  for i in range(Dataframe.shape[0]):
      if Dataframe.iloc[i - 1, Column] > 0:
          List_data_mean.append(Dataframe.iloc[i - 1, Column])

  Mean_list:int = int(np.mean(List_data_mean))
  return Mean_list
 
# ? Clean Mini-MIAS CSV

@Utilities.timer_func
def mini_mias_csv_clean(Dataframe:pd.DataFrame) -> pd.DataFrame:
  """
  Clean the data from the Mini-MIAS dataframe.

  Args:
      Dataframe (pd.DataFrame): Dataframe from Mini-MIAS' website.

  Returns:
      pd.DataFrame: Return the clean dataframe to use.
  """

  Value_fillna = 0
  # * This function will clean the data from the CSV archive

  Columns_list = ["REFNUM", "BG", "CLASS", "SEVERITY", "X", "Y", "RADIUS"]
  Dataframe_Mini_MIAS = pd.read_csv(Dataframe, usecols = Columns_list)

  # * Severity's column
  Mini_MIAS_severity_column:int = 3

  # * it labels each severity grade

  LE = LabelEncoder()
  Dataframe_Mini_MIAS.iloc[:, Mini_MIAS_severity_column].values
  Dataframe_Mini_MIAS.iloc[:, Mini_MIAS_severity_column] = LE.fit_transform(Dataframe_Mini_MIAS.iloc[:, 3])

  # * Fullfill X, Y and RADIUS columns with 0
  Dataframe_Mini_MIAS['X'] = Dataframe_Mini_MIAS['X'].fillna(Value_fillna)
  Dataframe_Mini_MIAS['Y'] = Dataframe_Mini_MIAS['Y'].fillna(Value_fillna)
  Dataframe_Mini_MIAS['RADIUS'] = Dataframe_Mini_MIAS['RADIUS'].fillna(Value_fillna)

  #Dataframe["X"].replace({"*NOTE": 0}, inplace = True)
  #Dataframe["Y"].replace({"3*": 0}, inplace = True)

  # * X and Y columns tranform into int type
  Dataframe_Mini_MIAS['X'] = Dataframe_Mini_MIAS['X'].astype(int)
  Dataframe_Mini_MIAS['Y'] = Dataframe_Mini_MIAS['Y'].astype(int)

  # * Severity and radius columns tranform into int type
  Dataframe_Mini_MIAS['SEVERITY'] = Dataframe_Mini_MIAS['SEVERITY'].astype(int)
  Dataframe_Mini_MIAS['RADIUS'] = Dataframe_Mini_MIAS['RADIUS'].astype(int)

  return Dataframe_Mini_MIAS

# ? Kmeans algorithm
@Utilities.timer_func
def kmeans_function(Folder_CSV: str, Folder_graph: str, Technique_name: str, X_data, Clusters: int, Filename: str, Severity: str) -> pd.DataFrame:
  """
  _summary_

  _extended_summary_

  Args:
      Folder_CSV (str): _description_
      Folder_graph (str): _description_
      Technique_name (str): _description_
      X_data (_type_): _description_
      Clusters (int): _description_
      Filename (str): _description_
      Severity (str): _description_

  Returns:
      pd.DataFrame: _description_
  """

  # * Tuple with different colors
  List_wcss = []
  Colors = ('red', 'blue', 'green', 'cyan', 'magenta', 'indigo', 'azure', 'tan', 'purple')

  for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_data)
    List_wcss.append(kmeans.inertia_)

  plt.plot(range(1, 10), List_wcss)
  plt.title('The Elbow Method')
  plt.xlabel('Number of clusters')
  plt.ylabel('WCSS')
  #plt.show()

  kmeans = KMeans(n_clusters = Clusters, init = 'k-means++', random_state = 42)
  y_kmeans = kmeans.fit_predict(X_data)

  for i in range(Clusters):

    if  Clusters <= 10:

        plt.scatter(X_data[y_kmeans == i, 0], X_data[y_kmeans == i, 1], s = 100, c = Colors[i], label = 'Cluster ' + str(i))


  plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroids')

  plt.title('Clusters')
  plt.xlabel('')
  plt.ylabel('')
  plt.legend()

  # * Tuple with different colors
  Folder_graph_name = 'Kmeans_Graph_{}_{}.png'.format(Technique_name, Severity)
  Folder_graph_folder = os.path.join(Folder_graph, Folder_graph_name)
  plt.savefig(Folder_graph_folder)
  #plt.show()

  DataFrame = pd.DataFrame({'y_kmeans' : y_kmeans, 'REFNUM' : Filename})
  #pd.set_option('display.max_rows', DataFrame.shape[0] + 1)
  #print(DataFrame)

  #pd.set_option('display.max_rows', DataFrame.shape[0] + 1)
  Dataframe_name = '{}_Dataframe_{}'.format(Technique_name, Severity)
  Dataframe_folder = os.path.join(Folder_CSV, Dataframe_name)

  DataFrame.to_csv(Dataframe_folder)

  #print(DataFrame['y_kmeans'].value_counts())

  return DataFrame

  # ? Remove Data from K-means function
  
@Utilities.timer_func
def kmeans_remove_data(Folder_path: str, Folder_CSV: str, Technique_name: str, Dataframe: pd.DataFrame, Cluster_to_remove: int, Severity: str) -> pd.DataFrame:
  """
  _summary_

  _extended_summary_

  Args:
      Folder_path (str): _description_
      Folder_CSV (str): _description_
      Technique_name (str): _description_
      Dataframe (pd.DataFrame): _description_
      Cluster_to_remove (int): _description_
      Severity (str): _description_

  Raises:
      ValueError: _description_

  Returns:
      pd.DataFrame: _description_
  """

  # * General lists
  #Images = [] # Png Images
  All_filename = [] 

  DataRemove = []
  Data = 0

  KmeansValue = 0
  Refnum = 1
  count = 1
  Index = 1

  os.chdir(Folder_path)

  # * Using sort function
  sorted_files, images = sort_images(Folder_path)

  # * Reading the files
  for File in sorted_files:

    Filename, Format = os.path.splitext(File)

    if Dataframe.iloc[Index - 1, Refnum] == Filename: # Read png files

      print(Filename)
      print(Dataframe.iloc[Index - 1, Refnum])

      if Dataframe.iloc[Index - 1, KmeansValue] == Cluster_to_remove:

        try:
          print(f"Working with {count} of {images} {Format} images, {Filename} ------- {Format} ✅")
          count += 1

          Path_File = os.path.join(Folder_path, File)
          os.remove(Path_File)
          print(Dataframe.iloc[Index - 1, Refnum], ' removed ❌')
          DataRemove.append(count)
          Data += 0

          #df = df.drop(df.index[count])

        except OSError:
          print('Cannot convert %s ❌' % File)

      elif Dataframe.iloc[Index - 1, KmeansValue] != Cluster_to_remove:
      
        All_filename.append(Filename)

      Index += 1

    elif Dataframe.iloc[Index - 1, Refnum] != Filename:
    
      print(Dataframe.iloc[Index - 1, Refnum]  + '----' + Filename)
      print(Dataframe.iloc[Index - 1, Refnum])
      raise ValueError("Files are not the same") #! Alert

    else:

      Index += 1

    for i in range(Data):

      Dataframe = Dataframe.drop(Dataframe.index[DataRemove[i]])

#Dataset = pd.DataFrame({'y_kmeans':df_u.iloc[Index - 1, REFNUM], 'REFNUM':df_u.iloc[Index - 1, KmeansValue]})
#X = Dataset.iloc[:, [0, 1, 2, 3, 4]].values

  #print(df)
  #pd.set_option('display.max_rows', df.shape[0] + 1)

  #Dataframe_name = str(Technique_name) + '_Data_Removed_' + str(Severity) + '.csv'
  Dataframe_name = '{}_Data_Removed_{}.csv'.format(Technique_name, Severity)
  Dataframe_folder = os.path.join(Folder_CSV, Dataframe_name)

  Dataframe.to_csv(Dataframe_folder)

  return Dataframe

# ?

class ChangeFormat(Utilities):
    """
    _summary_

    _extended_summary_

    Raises:
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
    """
    # * Change the format of one image to another 

    def __init__(self, **kwargs):
        """
        _summary_

        _extended_summary_

        Raises:
            ValueError: _description_
            TypeError: _description_
            ValueError: _description_
            TypeError: _description_
            ValueError: _description_
            TypeError: _description_
            ValueError: _description_
            TypeError: _description_
        """
        # * General parameters
        self.__Folder = kwargs.get('Folder', None)
        self.__New_folder = kwargs.get('Newfolder', None)
        self.__Format = kwargs.get('Format', None)
        self.__New_format = kwargs.get('Newformat', None)

        # * Values, type errors.
        if self.__Folder == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self.__Folder, str):
            raise TypeError("Folder attribute must be a string") #! Alert

        if self.__New_folder == None:
            raise ValueError("Folder destination does not exist") #! Alert
        if not isinstance(self.__New_folder, str):
            raise TypeError("Folder destination attribute must be a string") #! Alert

        if self.__Format == None:
            raise ValueError("Current format does not exist") #! Alert
        if not isinstance(self.__Format, str):
            raise TypeError("Current format must be a string") #! Alert

        if self.__New_format == None:
            raise ValueError("New format does not exist") #! Alert
        if not isinstance(self.__New_format, str):
            raise TypeError("Current format must be a string") #! Alert

    # * Folder attribute
    @property
    def Folder_property(self):
        return self.__Folder

    @Folder_property.setter
    def Folder_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Folder must be a string") #! Alert
        self.__Folder = New_value
    
    @Folder_property.deleter
    def Folder_property(self):
        print("Deleting folder...")
        del self.__Folder

    # * New folder attribute
    @property
    def New_folder_property(self):
        return self.__New_folder

    @New_folder_property.setter
    def New_folder_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Folder must be a string") #! Alert
        self.__New_folder = New_value
    
    @New_folder_property.deleter
    def New_folder_property(self):
        print("Deleting folder...")
        del self.__New_folder

    # * Format attribute
    @property
    def Format_property(self):
        return self.__Format

    @Format_property.setter
    def Format_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Format must be a string") #! Alert
        self.__Format = New_value
    
    @Format_property.deleter
    def New_folder_property(self):
        print("Deleting folder...")
        del self.__Format

    # * New Format attribute
    @property
    def New_format_property(self):
        return self.__New_format

    @New_format_property.setter
    def New_format_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("New format must be a string") #! Alert
        self.__New_format = New_value
    
    @New_format_property.deleter
    def New_format_property(self):
        print("Deleting new format...")
        del self.__New_format

    @Utilities.timer_func
    def ChangeFormat(self):
        """
        _summary_

        _extended_summary_
        """
        # * Changes the current working directory to the given path
        os.chdir(self.Folder)
        print(os.getcwd())
        print("\n")

        # * Using the sort function
        Sorted_files, Total_images = sort_images(self.__Folder)
        Count:int = 0

        # * Reading the files
        for File in Sorted_files:
            if File.endswith(self.__Format):

                try:
                    Filename, Format  = os.path.splitext(File)
                    print('Working with {} of {} {} images, {} ------- {} ✅'.format(Count, Total_images, self.__Format, Filename, self.__New_format))
                    #print(f"Working with {Count} of {Total_images} {self.Format} images, {Filename} ------- {self.New_format} ✅")
                    
                    # * Reading each image using cv2
                    Path_file = os.path.join(self.__Folder, File)
                    Image = cv2.imread(Path_file)         
                    #Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)
                    
                    # * Changing its format to a new one
                    New_name_filename = Filename + self.__New_format
                    New_folder = os.path.join(self.__New_folder, New_name_filename)

                    cv2.imwrite(New_folder, Image)
                    #FilenamesREFNUM.append(Filename)
                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert
                    #print('Cannot convert %s ❌' % File) #! Alert

        print("\n")
        #print(f"COMPLETE {Count} of {Total_images} TRANSFORMED ✅")
        print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

# ? Class for images cropping.

class CropImages(Utilities):
    """
    _summary_

    _extended_summary_
    """
    def __init__(self, **kwargs) -> None:
    
        """
        _summary_

        _extended_summary_

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        # * This algorithm outputs crop values for images based on the coordinates of the CSV file.
        # * General parameters
        self.__Folder: str = kwargs.get('Folder', None)
        self.__Normalfolder: str = kwargs.get('Normalfolder', None)
        self.__Tumorfolder: str = kwargs.get('Tumorfolder', None)
        self.__Benignfolder: str = kwargs.get('Benignfolder', None)
        self.__Malignantfolder: str = kwargs.get('Malignantfolder', None)

        # * CSV to extract data
        self.__Dataframe: pd.DataFrame = kwargs.get('Dataframe', None)
        self.__Shapes = kwargs.get('Shapes', None)
        
        # * X and Y mean to extract normal cropped images
        self.__X_mean:int = kwargs.get('Xmean', None)
        self.__Y_mean:int = kwargs.get('Ymean', None)

        if self.__Folder == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self.__Folder, str):
            raise TypeError("Folder must be a string") #! Alert

        if self.__Normalfolder == None:
            raise ValueError("Folder for normal images does not exist") #! Alert
        if not isinstance(self.__Normalfolder , str):
            raise TypeError("Folder must be a string") #! Alert

        if self.__Tumorfolder == None:
            raise ValueError("Folder for tumor images does not exist") #! Alert
        if not isinstance(self.__Tumorfolder, str):
            raise TypeError("Folder must be a string") #! Alert

        if self.__Benignfolder == None:
            raise ValueError("Folder for benign images does not exist") #! Alert
        if not isinstance(self.__Benignfolder, str):
            raise TypeError("Folder must be a string") #! Alert

        if self.__Malignantfolder == None:
            raise ValueError("Folder for malignant images does not exist") #! Alert
        if not isinstance(self.__Malignantfolder, str):
            raise TypeError("Folder must be a string") #! Alert
        
        #elif self.Dataframe == None:
        #raise ValueError("The dataframe is required") #! Alert

        elif self.__Shapes == None:
            raise ValueError("The shape is required") #! Alert

        elif self.__X_mean == None:
            raise ValueError("X_mean is required") #! Alert

        elif self.__Y_mean == None:
            raise ValueError("Y_mean is required") #! Alert

    @Utilities.timer_func
    def CropMIAS(self):
        
        #Images = []

        os.chdir(self.__Folder)

        # * Columns
        Name_column = 0
        Severity = 3
        X_column = 4
        Y_column = 5
        Radius = 6

        # * Labels
        Benign = 0
        Malignant = 1
        Normal = 2

        # * Initial index
        Index = 1
        
        # * Using sort function
        Sorted_files, Total_images = sort_images(self.__Folder)
        Count = 1

        # * Reading the files
        for File in Sorted_files:
        
            Filename, Format = os.path.splitext(File)

            print("******************************************")
            print(self.__Dataframe.iloc[Index - 1, Name_column])
            print(Filename)
            print("******************************************")

            if self.__Dataframe.iloc[Index - 1, Severity] == Benign:
                if self.__Dataframe.iloc[Index - 1, X_column] > 0  or self.__Dataframe.iloc[Index - 1, Y_column] > 0:
                
                    try:
                    
                        print(f"Working with {Count} of {Total_images} {Format} Benign images, {Filename} X: {self.__Dataframe.iloc[Index - 1, X_column]} Y: {self.__Dataframe.iloc[Index - 1, Y_column]}")
                        print(self.__Dataframe.iloc[Index - 1, Name_column], " ------ ", Filename, " ✅")
                        Count += 1

                        # * Reading the image
                        Path_file = os.path.join(self.__Folder, File)
                        Image = cv2.imread(Path_file)
                        
                        #Distance = self.Shape # X and Y.
                        #Distance = self.Shape # Perimetro de X y Y de la imagen.
                        #Image_center = Distance / 2 
                            
                        # * Obtaining the center using the radius
                        Image_center = self.__Dataframe.iloc[Index - 1, Radius] / 2 
                        # * Obtaining dimension
                        Height_Y = Image.shape[0] 
                        print(Image.shape[0])
                        print(self.__Dataframe.iloc[Index - 1, Radius])

                        # * Extract the value of X and Y of each image
                        X_size = self.__Dataframe.iloc[Index - 1, X_column]
                        print(X_size)
                        Y_size = self.__Dataframe.iloc[Index - 1, Y_column]
                        print(Y_size)
                            
                        # * Extract the value of X and Y of each image
                        XDL = X_size - Image_center
                        XDM = X_size + Image_center
                            
                        # * Extract the value of X and Y of each image
                        YDL = Height_Y - Y_size - Image_center
                        YDM = Height_Y - Y_size + Image_center

                        # * Cropped image
                        Cropped_Image_Benig = Image[int(YDL):int(YDM), int(XDL):int(XDM)]

                        print(Image.shape, " ----------> ", Cropped_Image_Benig.shape)

                        # print(Cropped_Image_Benig.shape)
                        # Display cropped image
                        # cv2_imshow(cropped_image)

                        New_name_filename = Filename + '_Benign_cropped' + Format

                        New_folder = os.path.join(self.__Benignfolder, New_name_filename)
                        cv2.imwrite(New_folder, Cropped_Image_Benig)

                        New_folder = os.path.join(self.__Tumorfolder, New_name_filename)
                        cv2.imwrite(New_folder, Cropped_Image_Benig)

                        #Images.append(Cropped_Image_Benig)

                    except OSError:
                            print('Cannot convert %s' % File)

            elif self.__Dataframe.iloc[Index - 1, Severity] == Malignant:
                if self.__Dataframe.iloc[Index - 1, X_column] > 0  or self.__Dataframe.iloc[Index - 1, Y_column] > 0:

                    try:

                        print(f"Working with {Count} of {Total_images} {Format} Malignant images, {Filename} X {self.__Dataframe.iloc[Index - 1, X_column]} Y {self.__Dataframe.iloc[Index - 1, Y_column]}")
                        print(self.__Dataframe.iloc[Index - 1, Name_column], " ------ ", Filename, " ✅")
                        Count += 1

                        # * Reading the image
                        Path_file = os.path.join(self.__Folder, File)
                        Image = cv2.imread(Path_file)
                        
                        #Distance = self.Shape # X and Y.
                        #Distance = self.Shape # Perimetro de X y Y de la imagen.
                        #Image_center = Distance / 2 
                            
                        # * Obtaining the center using the radius
                        Image_center = self.__Dataframe.iloc[Index - 1, Radius] / 2 # Center
                        # * Obtaining dimension
                        Height_Y = Image.shape[0] 
                        print(Image.shape[0])

                        # * Extract the value of X and Y of each image
                        X_size = self.__Dataframe.iloc[Index - 1, X_column]
                        Y_size = self.__Dataframe.iloc[Index - 1, Y_column]
                            
                        # * Extract the value of X and Y of each image
                        XDL = X_size - Image_center
                        XDM = X_size + Image_center
                            
                        # * Extract the value of X and Y of each image
                        YDL = Height_Y - Y_size - Image_center
                        YDM = Height_Y - Y_size + Image_center

                        # * Cropped image
                        Cropped_Image_Malig = Image[int(YDL):int(YDM), int(XDL):int(XDM)]

                        print(Image.shape, " ----------> ", Cropped_Image_Malig.shape)
                
                        # print(Cropped_Image_Malig.shape)
                        # Display cropped image
                        # cv2_imshow(cropped_image)

                        New_name_filename = Filename + '_Malignant_cropped' + Format

                        New_folder = os.path.join(self.__Malignantfolder, New_name_filename)
                        cv2.imwrite(New_folder, Cropped_Image_Malig)

                        New_folder = os.path.join(self.__Tumorfolder, New_name_filename)
                        cv2.imwrite(New_folder, Cropped_Image_Malig)

                        #Images.append(Cropped_Image_Malig)


                    except OSError:
                        print('Cannot convert %s' % File)
            
            elif self.__Dataframe.iloc[Index - 1, Severity] == Normal:
                if self.__Dataframe.iloc[Index - 1, X_column] == 0  or self.__Dataframe.iloc[Index - 1, Y_column] == 0:

                    try:

                        print(f"Working with {Count} of {Total_images} {Format} Normal images, {Filename}")
                        print(self.__Dataframe.iloc[Index - 1, Name_column], " ------ ", Filename, " ✅")
                        Count += 1

                        Path_file = os.path.join(self.__Folder, File)
                        Image = cv2.imread(Path_file)

                        Distance = self.__Shapes # Perimetro de X y Y de la imagen.
                        Image_center = Distance / 2 # Centro de la imagen.
                        #CD = self.df.iloc[Index - 1, Radius] / 2
                        # * Obtaining dimension
                        Height_Y = Image.shape[0] 
                        print(Image.shape[0])

                        # * Extract the value of X and Y of each image
                        X_size = self.__X_mean
                        Y_size = self.__Y_mean
                            
                        # * Extract the value of X and Y of each image
                        XDL = X_size - Image_center
                        XDM = X_size + Image_center
                            
                        # * Extract the value of X and Y of each image
                        YDL = Height_Y - Y_size - Image_center
                        YDM = Height_Y - Y_size + Image_center

                        # * Cropped image
                        Cropped_Image_Normal = Image[int(YDL):int(YDM), int(XDL):int(XDM)]

                        # * Comparison two images
                        print(Image.shape, " ----------> ", Cropped_Image_Normal.shape)

                        # print(Cropped_Image_Normal.shape)
                        # Display cropped image
                        # cv2_imshow(cropped_image)
                    
                        New_name_filename = Filename + '_Normal_cropped' + Format

                        New_folder = os.path.join(self.__Normalfolder, New_name_filename)
                        cv2.imwrite(New_folder, Cropped_Image_Normal)

                        #Images.append(Cropped_Image_Normal)

                    except OSError:
                        print('Cannot convert %s' % File)

            Index += 1   

# ? ####################################################### CBIS-DDSM #######################################################

# ? CBIS-DDSM split data

@Utilities.timer_func
def CBIS_DDSM_split_data(**kwargs) -> pd.DataFrame:
    """
    _summary_

    _extended_summary_

    Args:
        Folder_CSV (_type_): _description_
        Folder (_type_): _description_
        Folder_total_benign (_type_): _description_
        Folder_benign (_type_): _description_
        Folder_benign_wc (_type_): _description_
        Folder_malignant (_type_): _description_
        Folder_abnormal (_type_): _description_
        Dataframe (_type_): _description_
        Severity (_type_): _description_
        Phase (_type_): _description_

    Returns:
        _type_: _description_
    """
    # * General parameters
    Folder = kwargs.get('folder', None)
    Folder_total_benign = kwargs.get('Allbenignfolder', None)
    Folder_benign = kwargs.get('benignfolder', None)
    Folder_benign_wc = kwargs.get('benignWCfolder', None)
    Folder_malignant = kwargs.get('malignantfolder', None)
    Folder_abnormal = kwargs.get('Abnormalfolder', None)
    Folder_CSV = kwargs.get('csvfolder', None)

    Dataframe = kwargs.get('dataframe', None)
    Severity = kwargs.get('severity', None)
    Stage = kwargs.get('stage', None)

    Save_file = kwargs.get('savefile', False)

    # * Folder attribute (ValueError, TypeError)
    if Folder == None:
      raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder, str):
      raise TypeError("Folder attribute must be a string") #! Alert

    # * Folder CSV (ValueError, TypeError)
    if Folder_CSV == None:
      raise ValueError("Folder to save csv does not exist") #! Alert
    if not isinstance(Folder_CSV, str):
      raise TypeError("Folder to save csv must be a string") #! Alert

    # * Folder benign (ValueError, TypeError)
    if Folder_benign == None:
      raise ValueError("Folder benign does not exist") #! Alert
    if not isinstance(Folder_benign, str):
      raise TypeError("Folder benign must be a string") #! Alert

    # * Folder benign without callback (ValueError, TypeError)
    if Folder_benign_wc == None:
      raise ValueError("Folder benign without callback does not exist") #! Alert
    if not isinstance(Folder_benign_wc, str):
      raise TypeError("Folder abenign without callback must be a string") #! Alert

    # * Folder malignant (ValueError, TypeError)
    if Folder_malignant == None:
      raise ValueError("Folder malignant does not exist") #! Alert
    if not isinstance(Folder_malignant, str):
      raise TypeError("Folder malignant must be a string") #! Alert

    # * Dataframe (ValueError, TypeError)
    if Dataframe == None:
      raise ValueError("Dataframe does not exist") #! Alert
    if not isinstance(Dataframe, pd.DataFrame):
      raise TypeError("Dataframe must be a dataframe") #! Alert

    # * Severity (ValueError, TypeError)
    if Severity == None:
      raise ValueError("Severity label does not exist") #! Alert
    if not isinstance(Severity, str):
      raise TypeError("Severity label must be a string") #! Alert

    # * Folder attribute (ValueError, TypeError)
    if Stage == None:
      raise ValueError("Stage does not exist") #! Alert
    if not isinstance(Stage, str):
      raise TypeError("Stage label must be a string") #! Alert

    # * Save_file (ValueError, TypeError)
    if (Folder_CSV != None and Save_file == True):
      if not isinstance(Folder_CSV, str):
        raise TypeError("Folder destination must be a string") #! Alert
    elif (Folder_CSV == None and Save_file == True):
      warnings.warn('Saving the images is available but a folder destination was not found') #! Alert
      print("\n")
    elif (Folder_CSV != None and Save_file == False):
      warnings.warn('Saving the images is unavailable but a folder destination was found') #! Alert
      print("\n")
    else:
      pass

    #remove_all_files(Folder_total_benign)
    #remove_all_files(Folder_benign)
    #remove_all_files(Folder_benign_wc)
    #remove_all_files(Folder_malignant)
    #remove_all_files(Folder_abnormal)
    
    # * Lists to save the images and their respective labels
    Images = []
    Label = []

    # * Lists to save filename of each severity
    Filename_benign_all = []
    Filename_malignant_all = []
    Filename_all = []
    Filename_benign_list = []
    Filename_benign_WC_list = []
    Filename_malignant_list = []
    Filename_abnormal_list = []
    
    # * Label for each severity
    Benign_label = 0
    Benign_without_callback_label = 1
    Malignant_label = 2

    # * String label for each severity
    Benign_label_string = 'Benign'
    Benign_without_callback_label_string = 'Benign without callback'
    Malignant_label_string = 'Malignant'

    # * Initial index
    Index = 0
    
    # * Initial count variable
    Count = 1

    # * Change the current directory to specified directory
    os.chdir(Folder)

    # * Sort images from the function
    Sorted_files, Total_images = sort_images(Folder)

    for File in Sorted_files:
        
        # * Extract filename and format for each file
        Filename, Format = os.path.splitext(File)

        if Dataframe[Index] == Benign_label:

                try:
                    
                    print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Benign_label_string), str(Filename)))
                    #print(f"Working with {Count} of {Total_images} Benign images, {Filename}")

                    # * Get the file from the folder
                    File_folder = os.path.join(Folder, File)
                    Image_benign = cv2.imread(File_folder)
                    
                    # * Combine name with the format
                    #Filename_benign = Filename + '_Benign'
                    Filename_benign = '{}_Benign'.format(str(Filename))
                    Filename_benign_format = Filename_benign + Format
                    
                    # * Save images in their folder, respectively
                    Filename_total_benign_folder = os.path.join(Folder_total_benign, Filename_benign_format)
                    cv2.imwrite(Filename_total_benign_folder, Image_benign)

                    Filename_abnormal_folder = os.path.join(Folder_abnormal, Filename_benign_format)
                    cv2.imwrite(Filename_abnormal_folder, Image_benign)

                    Filename_benign_folder = os.path.join(Folder_benign, Filename_benign_format)
                    cv2.imwrite(Filename_benign_folder, Image_benign)

                    # * Add data into the lists
                    Images.append(Image_benign)
                    Label.append(Benign_label)

                    Filename_benign_all.append(Filename_benign)
                    Filename_abnormal_list.append(Filename_benign)
                    Filename_benign_list.append(Filename_benign)
                    Filename_all.append(Filename_benign)
                    
                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert

        elif Dataframe[Index] == Benign_without_callback_label:
    
                try:

                    print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Benign_without_callback_label_string), str(Filename)))
                    #print(f"Working with {Count} of {Total_images} Benign images, {Filename}")
                    
                    # * Get the file from the folder
                    File_folder = os.path.join(Folder, File)
                    Image_benign_without_callback = cv2.imread(File_folder)

                    # * Combine name with the format
                    #Filename_benign_WC = Filename + '_Benign_Without_Callback'
                    Filename_benign_WC = '{}_Benign_Without_Callback'.format(str(Filename))
                    Filename_benign_WC_format = Filename_benign_WC + Format

                    # * Save images in their folder, respectively
                    Filename_total_benign_folder = os.path.join(Folder_total_benign, Filename_benign_WC_format)
                    cv2.imwrite(Filename_total_benign_folder, Image_benign_without_callback)

                    Filename_benign_folder = os.path.join(Folder_benign_wc, Filename_benign_WC_format)
                    cv2.imwrite(Filename_benign_folder, Image_benign_without_callback)

                    # * Add data into the lists
                    Images.append(Image_benign_without_callback)
                    Label.append(Benign_without_callback_label)

                    Filename_benign_all.append(Filename_benign_WC)
                    Filename_benign_WC_list.append(Filename_benign_WC)
                    Filename_all.append(Filename_benign_WC)

                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert
        
        elif Dataframe[Index] == Malignant_label:

                try:

                    print(f"Working with {Count} of {Total_images} Malignant images, {Filename}")
                    print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Malignant_label_string ), str(Filename)))
                    #print(f"Working with {Count} of {Total_images} Benign images, {Filename}")

                    # * Get the file from the folder
                    File_folder = os.path.join(Folder, File)
                    Image_malignant = cv2.imread(File_folder)

                    # * Combine name with the format
                    #Filename_malignant = Filename + '_Malignant'
                    Filename_malignant = '{}_Malignant'.format(str(Filename))
                    Filename_malignant_format = Filename_malignant + Format

                    # * Save images in their folder, respectively
                    Filename_abnormal_folder = os.path.join(Folder_abnormal, Filename_malignant_format)
                    cv2.imwrite(Filename_abnormal_folder, Image_malignant)

                    Filename_malignant_folder = os.path.join(Folder_malignant, Filename_malignant_format)
                    cv2.imwrite(Filename_malignant_folder, Image_malignant)

                    # * Add data into the lists
                    Images.append(Image_malignant)
                    Label.append(Malignant_label)

                    Filename_malignant_all.append(Filename_malignant)
                    Filename_abnormal_list.append(Filename_malignant)
                    Filename_malignant_list.append(Filename_malignant)
                    Filename_all.append(Filename_malignant)
                    
                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert
                    
        Index += 1

    Dataframe_labeled = pd.DataFrame({'Filenames':Filename_all,'Labels':Label}) 

    if Save_file == True:
        #Dataframe_labeled_name = 'CBIS_DDSM_Split_' + 'Dataframe_' + str(Severity) + '_' + str(Stage) + '.csv' 
        Dataframe_labeled_name = 'CBIS_DDSM_Split_Dataframe_{}_{}.csv'.format(str(Severity), str(Stage))
        Dataframe_labeled_folder = os.path.join(Folder_CSV, Dataframe_labeled_name)
        Dataframe_labeled.to_csv(Dataframe_labeled_folder)

    return Dataframe_labeled

# ? CBIS-DDSM split all data

@Utilities.timer_func
def CBIS_DDSM_split_several_data(**kwargs)-> pd.DataFrame:
    """
    _summary_

    _extended_summary_

    Args:
        Folder_CSV (_type_): _description_
        Folder_test (_type_): _description_
        Folder_training (_type_): _description_
        Folder_total_benign (_type_): _description_
        Folder_benign (_type_): _description_
        Folder_benign_wc (_type_): _description_
        Folder_malignant (_type_): _description_
        Folder_abnormal (_type_): _description_
        Dataframe_test (_type_): _description_
        Dataframe_training (_type_): _description_
        Severity (_type_): _description_
        Phase (_type_): _description_

    Returns:
        _type_: _description_
    """

    # * General parameters
    Folder_test = kwargs.get('testfolder', None)
    Folder_training = kwargs.get('trainingfolder', None)
    Folder_total_benign = kwargs.get('Allbenignfolder', None)
    Folder_benign = kwargs.get('benignfolder', None)
    Folder_benign_wc = kwargs.get('benignWCfolder', None)
    Folder_malignant = kwargs.get('malignantfolder', None)
    Folder_abnormal = kwargs.get('Abnormalfolder', None)
    Folder_CSV = kwargs.get('csvfolder', None)

    Dataframe_test = kwargs.get('dftest', None)
    Dataframe_training = kwargs.get('dftraining', None)
    Severity = kwargs.get('severity', None)
    Stage_test = kwargs.get('stage', None)
    Stage_training = kwargs.get('stage', None)

    Save_file = kwargs.get('savefile', False)

    #remove_all_files(Folder_total_benign)
    #remove_all_files(Folder_benign)
    #remove_all_files(Folder_benign_wc)
    #remove_all_files(Folder_malignant)
    #remove_all_files(Folder_abnormal)

    # * Folder test (ValueError, TypeError)
    if Folder_test == None:
      raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_test, str):
      raise TypeError("Folder attribute must be a string") #! Alert

    # * Folder training (ValueError, TypeError)
    if Folder_training == None:
      raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_training, str):
      raise TypeError("Folder attribute must be a string") #! Alert

    # * Folder CSV (ValueError, TypeError)
    if Folder_CSV == None:
      raise ValueError("Folder to save csv does not exist") #! Alert
    if not isinstance(Folder_CSV, str):
      raise TypeError("Folder to save csv must be a string") #! Alert

    # * Folder benign (ValueError, TypeError)
    if Folder_benign == None:
      raise ValueError("Folder benign does not exist") #! Alert
    if not isinstance(Folder_benign, str):
      raise TypeError("Folder benign must be a string") #! Alert

    # * Folder benign without callback (ValueError, TypeError)
    if Folder_benign_wc == None:
      raise ValueError("Folder benign without callback does not exist") #! Alert
    if not isinstance(Folder_benign_wc, str):
      raise TypeError("Folder abenign without callback must be a string") #! Alert

    # * Folder malignant (ValueError, TypeError)
    if Folder_malignant == None:
      raise ValueError("Folder malignant does not exist") #! Alert
    if not isinstance(Folder_malignant, str):
      raise TypeError("Folder malignant must be a string") #! Alert

    # * Dataframe (ValueError, TypeError)
    if Dataframe_test == None:
      raise ValueError("Dataframe test does not exist") #! Alert
    if not isinstance(Dataframe_test, pd.DataFrame):
      raise TypeError("Dataframe test must be a dataframe") #! Alert

    # * Dataframe (ValueError, TypeError)
    if Dataframe_training == None:
      raise ValueError("Dataframe training does not exist") #! Alert
    if not isinstance(Dataframe_training, pd.DataFrame):
      raise TypeError("Dataframe training must be a dataframe") #! Alert

    # * Severity (ValueError, TypeError)
    if Severity == None:
      raise ValueError("Severity label does not exist") #! Alert
    if not isinstance(Severity, str):
      raise TypeError("Severity label must be a string") #! Alert

    # * Stage test (ValueError, TypeError)
    if Stage_test == None:
      raise ValueError("Stage does not exist") #! Alert
    if not isinstance(Stage_test, str):
      raise TypeError("Stage label must be a string") #! Alert

    # * Stage training (ValueError, TypeError)
    if Stage_test == None:
      raise ValueError("Stage does not exist") #! Alert
    if not isinstance(Stage_test, str):
      raise TypeError("Stage label must be a string") #! Alert

    # * Save_file (ValueError, TypeError)
    if (Folder_CSV != None and Save_file == True):
      if not isinstance(Folder_CSV, str):
        raise TypeError("Folder destination must be a string") #! Alert
    elif (Folder_CSV == None and Save_file == True):
      warnings.warn('Saving the images is available but a folder destination was not found') #! Alert
      print("\n")
    elif (Folder_CSV != None and Save_file == False):
      warnings.warn('Saving the images is unavailable but a folder destination was found') #! Alert
      print("\n")
    else:
      pass

    # * Lists to save the images and their respective labels
    Images = []
    Label = []

    # * Lists to save filename of each severity
    Filename_benign_all = []
    Filename_malignant_all = []
    Filename_all = []

    Filename_benign_list = []
    Filename_benign_WC_list = []
    Filename_malignant_list = []
    Filename_abnormal_list = []
    
    # * Label for each severity
    Benign_label = 0
    Benign_without_callback_label = 1
    Malignant_label = 2

    # * String label for each severity
    Benign_label_string = 'Benign'
    Benign_without_callback_label_string = 'Benign without callback'
    Malignant_label_string = 'Malignant'

    # * Initial index
    Index = 0
    
    # * Initial count variable
    Count = 1

    # * Change the current directory to specified directory - Test
    os.chdir(Folder_test)

    # * Sort images from the function
    Sorted_files, Total_images = sort_images(Folder_test)
    
    for File in Sorted_files:
        
        Filename, Format  = os.path.splitext(File)
        
        if Dataframe_test[Index] == Benign_label:

            try:
                print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Benign_label_string), str(Filename)))
                #print(f"Working with {Count} of {Total_images} Benign images, {Filename}")

                # * Get the file from the folder
                File_folder = os.path.join(Folder_test, File)
                Image_benign = cv2.imread(File_folder)

                # * Combine name with the format
                #Filename_benign = Filename + '_benign'
                Filename_benign = '{}_Benign'.format(str(Filename))
                Filename_benign_format = Filename_benign + Format
                
                # * Save images in their folder, respectively
                Filename_abnormal_folder = os.path.join(Folder_abnormal, Filename_benign_format)
                cv2.imwrite(Filename_abnormal_folder, Image_benign)

                Filename_total_benign_folder = os.path.join(Folder_total_benign, Filename_benign_format)
                cv2.imwrite(Filename_total_benign_folder, Image_benign)

                Filename_benign_folder = os.path.join(Folder_benign, Filename_benign_format)
                cv2.imwrite(Filename_benign_folder, Image_benign)

                # * Add data into the lists
                Images.append(Image_benign)
                Label.append(Benign_label)

                Filename_benign_all.append(Filename_benign)
                Filename_abnormal_list.append(Filename_benign)
                Filename_benign_list.append(Filename_benign)
                Filename_all.append(Filename_benign)

                Count += 1

            except OSError:
                print('Cannot convert {} ❌'.format(str(File))) #! Alert

        elif Dataframe_test[Index] == Benign_without_callback_label:
    
            try:
                print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Benign_without_callback_label_string), str(Filename)))
                #print(f"Working with {Count} of {Total_images} Benign images, {Filename}")

                # * Get the file from the folder
                File_folder = os.path.join(Folder_test, File)
                Image_benign_without_callback = cv2.imread(File_folder)

                # * Combine name with the format
                #Filename_benign = Filename + '_benign'
                Filename_benign_WC = '{}_Benign_Without_Callback'.format(str(Filename))
                Filename_benign_WC_format = Filename_benign_WC + Format
                
                # * Save images in their folder, respectively
                Filename_total_benign_WC_folder = os.path.join(Folder_total_benign, Filename_benign_WC_format)
                cv2.imwrite(Filename_total_benign_WC_folder, Image_benign_without_callback)

                Filename_benign_WC_folder = os.path.join(Folder_benign, Filename_benign_WC_format)
                cv2.imwrite(Filename_benign_WC_folder, Image_benign_without_callback)

                # * Add data into the lists
                Images.append(Image_benign_without_callback)
                Label.append(Benign_without_callback_label)

                Filename_benign_all.append(Filename_benign_WC)
                Filename_benign_WC_list.append(Filename_benign_WC)
                Filename_all.append(Filename_benign_WC)

                Count += 1

            except OSError:
                print('Cannot convert {} ❌'.format(str(File))) #! Alert
        
        elif Dataframe_test[Index] == Malignant_label:

                try:
                    #print(f"Working with {Count} of {Total_images} Malignant images, {Filename}")
                    print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Malignant_label_string), str(Filename)))

                    # * Get the file from the folder
                    File_folder = os.path.join(Folder_test, File)
                    Image_malignant = cv2.imread(File_folder)

                    # * Combine name with the format
                    #Filename_malignant = Filename + '_Malignant'
                    Filename_malignant = '{}_Malignant'.format(str(Filename))
                    Filename_malignant_format = Filename_malignant + Format

                    # * Save images in their folder, respectively
                    Filename_abnormal_folder = os.path.join(Folder_abnormal, Filename_malignant_format)
                    cv2.imwrite(Filename_abnormal_folder, Image_malignant)

                    Filename_malignant_folder = os.path.join(Folder_malignant, Filename_malignant_format)
                    cv2.imwrite(Filename_malignant_folder, Image_malignant)

                    # * Add data into the lists
                    Images.append(Image_malignant)
                    Label.append(Malignant_label)

                    Filename_malignant_all.append(Filename_malignant)
                    Filename_abnormal_list.append(Filename_malignant)
                    Filename_malignant_list.append(Filename_malignant)
                    Filename_all.append(Filename_malignant)
                    
                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert

        Index += 1
        #print(Index)
    
    # * Change the current directory to specified directory - Training
    os.chdir(Folder_training)
    
    # * Change the variable's value
    Index = 0
    count = 1

    # * Sort images from the function
    Sorted_files, Total_images = sort_images(Folder_training)
    
    for File in Sorted_files:
        
        Filename, Format  = os.path.splitext(File)
        
        if Dataframe_test[Index] == Benign_label:

            try:
                print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Benign_label_string), str(Filename)))
                #print(f"Working with {Count} of {Total_images} Benign images, {Filename}")

                # * Get the file from the folder
                File_folder = os.path.join(Folder_test, File)
                Image_benign = cv2.imread(File_folder)

                # * Combine name with the format
                #Filename_benign = Filename + '_benign'
                Filename_benign = '{}_Benign'.format(str(Filename))
                Filename_benign_format = Filename_benign + Format
                
                # * Save images in their folder, respectively
                Filename_abnormal_folder = os.path.join(Folder_abnormal, Filename_benign_format)
                cv2.imwrite(Filename_abnormal_folder, Image_benign)

                Filename_total_benign_folder = os.path.join(Folder_total_benign, Filename_benign_format)
                cv2.imwrite(Filename_total_benign_folder, Image_benign)

                Filename_benign_folder = os.path.join(Folder_benign, Filename_benign_format)
                cv2.imwrite(Filename_benign_folder, Image_benign)

                # * Add data into the lists
                Images.append(Image_benign)
                Label.append(Benign_label)

                Filename_benign_all.append(Filename_benign)
                Filename_abnormal_list.append(Filename_benign)
                Filename_benign_list.append(Filename_benign)
                Filename_all.append(Filename_benign)

                Count += 1

            except OSError:
                print('Cannot convert {} ❌'.format(str(File))) #! Alert

        elif Dataframe_test[Index] == Benign_without_callback_label:
    
            try:
                print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Benign_without_callback_label_string), str(Filename)))
                #print(f"Working with {Count} of {Total_images} Benign images, {Filename}")

                # * Get the file from the folder
                File_folder = os.path.join(Folder_test, File)
                Image_benign_without_callback = cv2.imread(File_folder)

                # * Combine name with the format
                #Filename_benign = Filename + '_benign'
                Filename_benign_WC = '{}_Benign_Without_Callback'.format(str(Filename))
                Filename_benign_WC_format = Filename_benign_WC + Format
                
                # * Save images in their folder, respectively
                Filename_total_benign_WC_folder = os.path.join(Folder_total_benign, Filename_benign_WC_format)
                cv2.imwrite(Filename_total_benign_WC_folder, Image_benign_without_callback)

                Filename_benign_WC_folder = os.path.join(Folder_benign, Filename_benign_WC_format)
                cv2.imwrite(Filename_benign_WC_folder, Image_benign_without_callback)

                # * Add data into the lists
                Images.append(Image_benign_without_callback)
                Label.append(Benign_without_callback_label)

                Filename_benign_all.append(Filename_benign_WC)
                Filename_benign_WC_list.append(Filename_benign_WC)
                Filename_all.append(Filename_benign_WC)

                Count += 1

            except OSError:
                print('Cannot convert {} ❌'.format(str(File))) #! Alert
        
        elif Dataframe_test[Index] == Malignant_label:

                try:
                    #print(f"Working with {Count} of {Total_images} Malignant images, {Filename}")
                    print('Working with {} of {} {} images {} ✅'.format(str(Count), str(Total_images), str(Malignant_label_string), str(Filename)))

                    # * Get the file from the folder
                    File_folder = os.path.join(Folder_test, File)
                    Image_malignant = cv2.imread(File_folder)

                    # * Combine name with the format
                    #Filename_malignant = Filename + '_Malignant'
                    Filename_malignant = '{}_Malignant'.format(str(Filename))
                    Filename_malignant_format = Filename_malignant + Format

                    # * Save images in their folder, respectively
                    Filename_abnormal_folder = os.path.join(Folder_abnormal, Filename_malignant_format)
                    cv2.imwrite(Filename_abnormal_folder, Image_malignant)

                    Filename_malignant_folder = os.path.join(Folder_malignant, Filename_malignant_format)
                    cv2.imwrite(Filename_malignant_folder, Image_malignant)

                    # * Add data into the lists
                    Images.append(Image_malignant)
                    Label.append(Malignant_label)

                    Filename_malignant_all.append(Filename_malignant)
                    Filename_abnormal_list.append(Filename_malignant)
                    Filename_malignant_list.append(Filename_malignant)
                    Filename_all.append(Filename_malignant)
                    
                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert

        Index += 1

    Dataframe_labeled = pd.DataFrame({'Filenames':Filename_all,'Labels':Label}) 

    if Save_file == True:
        #Dataframe_labeled_name = 'CBIS_DDSM_Split_' + 'Dataframe_' + str(Severity) + '_' + str(Stage) + '.csv' 
        Dataframe_labeled_name = 'CBIS_DDSM_Split_Dataframe_{}_{}_{}.csv'.format(str(Severity), str(Stage_test), str(Stage_training))
        Dataframe_labeled_folder = os.path.join(Folder_CSV, Dataframe_labeled_name)
        Dataframe_labeled.to_csv(Dataframe_labeled_folder)

    return Dataframe_labeled

# ? Dataset splitting

def Dataframe_split(Dataframe: pd.DataFrame) -> tuple[pd.DataFrame, set, set, set]:
  """
  _summary_

  _extended_summary_

  Args:
      Dataframe (pd.DataFrame): _description_

  Returns:
      tuple[pd.DataFrame, set, set, set]: _description_
  """
  X = Dataframe.drop('Labels', axis = 1)
  Y = Dataframe['Labels']

  Majority, Minority = Dataframe['Labels'].value_counts()

  return X, Y, Majority, Minority

# ? Imbalance data majority

def Imbalance_data_majority(Dataframe: pd.DataFrame) -> tuple[pd.DataFrame, set, set, set]:
    """
    _summary_

    _extended_summary_

    Args:
        Dataframe (pd.DataFrame): _description_

    Returns:
        tuple[pd.DataFrame, set, set, set]: _description_
    """
    X = Dataframe.drop('Labels', axis = 1)
    Y = Dataframe['Labels']

    # * Value counts from Y(Labels)
    Majority, Minority = Y.value_counts()

    Dataframe_majority = Dataframe[Y == 0]
    Dataframe_minority = Dataframe[Y == 1]
    
    # * Resample using majority
    Dataframe_majority_downsampled = resample(  Dataframe_majority, 
                                                replace = False,        # sample with replacement
                                                n_samples = Minority,   # to match majority class
                                                random_state = 123)     # reproducible results
    
    # * Concat minority and majority downsampled
    Dataframe_downsampled = pd.concat([Dataframe_minority, Dataframe_majority_downsampled])
    print(Dataframe_downsampled['Labels'].value_counts())

    X = Dataframe_downsampled.drop('Labels', axis = 1)
    Y = Dataframe_downsampled['Labels']

    return X, Y, Majority, Minority

# ? Imbalance data minority

def Imbalance_data_minority(Dataframe: pd.DataFrame) -> tuple[pd.DataFrame, set, set, set]:
    """
    _summary_

    _extended_summary_

    Args:
        Dataframe (pd.DataFrame): _description_

    Returns:
        tuple[pd.DataFrame, set, set, set]: _description_
    """
    X = Dataframe.drop('Labels', axis = 1)
    Y = Dataframe['Labels']
    
    # * Value counts from Y(Labels)
    Majority, Minority = Y.value_counts()

    Dataframe_majority = Dataframe[Y == 0]
    Dataframe_minority = Dataframe[Y == 1]

    # * Resample using minority
    Dataframe_minority_upsampled = resample(    Dataframe_minority, 
                                                replace = True,         # sample with replacement
                                                n_samples = Majority,   # to match majority class
                                                random_state = 123)     # reproducible results

    # * Concat majority and minority upsampled
    Dataframe_upsampled = pd.concat([Dataframe_majority, Dataframe_minority_upsampled])
    print(Dataframe_upsampled['Labels'].value_counts())

    X = Dataframe_upsampled.drop('Labels', axis = 1)
    Y = Dataframe_upsampled['Labels']

    return X, Y, Majority, Minority

# ? Convertion severity to int value

@Utilities.timer_func
def CBIS_DDSM_CSV_severity_labeled(Folder_CSV: str, Column: int, Severity: int)-> pd.DataFrame:
    """
    _summary_

    _extended_summary_

    Args:
        Folder_CSV(str): _description_
        Column(int): _description_
        Severity(int): _description_

    Returns:
        pd.DataFrame: _description_
    """

    # * Folder attribute (ValueError, TypeError)
    if Folder_CSV == None:
        raise ValueError("Folder csv does not exist") #! Alert
    if not isinstance(Folder_CSV, str):
        raise TypeError("Folder csv must be a string") #! Alert

    # * Folder attribute (ValueError, TypeError)
    if Column == None:
        raise ValueError("Column does not exist") #! Alert
    if not isinstance(Column, int):
        raise TypeError("Column must be a integer") #! Alert

    # * Folder attribute (ValueError, TypeError)
    if Severity == None:
        raise ValueError("Severity does not exist") #! Alert
    if Severity >= 3:
        raise ValueError("Severity must be less than 3") #! Alert
    if not isinstance(Severity, int):
        raise TypeError("Severity must be a integer") #! Alert

    # * Label encoder class
    LE = LabelEncoder()

    # * Severity labels
    Calcification = 1
    Mass = 2

    # * Dataframe headers
    if Severity == Calcification:

        Columns_list = ["patient_id", "breast density", "left or right breast", "image view", 
                        "abnormality id", "abnormality type", "calc type", "calc distribution", 
                        "assessment", "pathology", "subtlety", "image file path", "cropped image file path", 
                        "ROI mask file path"]
    if Severity == Mass:

        Columns_list = ["patient_id", "breast_density", "left or right breast", "image view", 
                        "abnormality id", "abnormality type", "mass shape", "mass margins", 
                        "assessment", "pathology", "subtlety", "image file path", "cropped image file path", 
                        "ROI mask file path"]
    
    # * Dataframe headers between calfications or masses
    Dataframe_severity = pd.read_csv(Folder_CSV, usecols = Columns_list)

    # * Getting the values and label them
    Dataframe_severity.iloc[:, Column].values
    Dataframe_severity.iloc[:, Column] = LE.fit_transform(Dataframe_severity.iloc[:, Column])

    Dataset_severity_labeled = Dataframe_severity.iloc[:, Column].values
    Dataframe = Dataframe_severity.iloc[:, Column]

    print(Dataset_severity_labeled)
    pd.set_option('display.max_rows', Dataframe.shape[0] + 1)
    print(Dataframe.value_counts())

    return Dataset_severity_labeled

# ? Concat multiple dataframes
@Utilities.timer_func
def concat_dataframe(*dfs: pd.DataFrame, **kwargs: str) -> pd.DataFrame:
  """
  Concat multiple dataframes and name it using technique and the class problem

  Args:
      Dataframe (pd.DataFrames): Multiple dataframes can be entered for concatenation

  Raises:
      ValueError: If the folder variable does not found give a error
      TypeError: _description_
      Warning: _description_
      TypeError: _description_
      Warning: _description_
      TypeError: _description_

  Returns:
      pd.DataFrame: Return the concatenated dataframe
  """
  # * this function concatenate the number of dataframes added

  # * General parameters

  Folder_path = kwargs.get('folder', None)
  Technique = kwargs.get('technique', None)
  Class_problem = kwargs.get('classp', None)
  Save_file = kwargs.get('savefile', False)

  # * Values, type errors and warnings
  if Folder_path == None:
    raise ValueError("Folder does not exist") #! Alert
  if not isinstance(Folder_path, str):
    raise TypeError("Folder attribute must be a string") #! Alert
  
  if Technique == None:
    #raise ValueError("Technique does not exist")  #! Alert
    warnings.warn("Technique does not found, the string 'Without_Technique' will be implemented") #! Alert
    Technique = 'Without_Technique'
  if not isinstance(Technique, str):
    raise TypeError("Technique attribute must be a string") #! Alert

  if Class_problem == None:
    #raise ValueError("Class problem does not exist")  #! Alert
    warnings.warn("Class problem does not found, the string 'No_Class' will be implemented") #! Alert
  if not isinstance(Class_problem, str):
    raise TypeError("Class problem must be a string") #! Alert

  # * Concatenate each dataframe
  ALL_dataframes = [df for df in dfs]
  print(len(ALL_dataframes))
  Final_dataframe = pd.concat(ALL_dataframes, ignore_index = True, sort = False)
      
  #pd.set_option('display.max_rows', Final_dataframe.shape[0] + 1)
  #print(DataFrame)

  # * Name the final dataframe and save it into the given path

  if Save_file == True:
    #Name_dataframe =  str(Class_problem) + '_Dataframe_' + str(Technique) + '.csv'
    Dataframe_name = '{}_Dataframe_{}.csv'.format(str(Class_problem), str(Technique))
    Dataframe_folder_save = os.path.join(Folder_path, Dataframe_name)
    Final_dataframe.to_csv(Dataframe_folder_save)

  return Final_dataframe

# ? Split folders into train/test/validation

class SplitDataFolder(Utilities):

    # * Change the format of one image to another 

    def __init__(self, **kwargs):

        # * General parameters
        self.__Folder = kwargs.get('Folder', None)
    
    # * Folder attribute
    @property
    def __Folder_property(self):
        return self.__Folder

    @__Folder_property.setter
    def __Folder_property(self, New_value):
        self.__Folder = New_value
    
    @__Folder_property.deleter
    def __Folder_property(self):
        print("Deleting folder...")
        del self.__Folder

    @Utilities.timer_func
    def split_folders_train_test_val(self) -> str:
        """
        Create a new folder with the folders of the class problem and its distribution of training, test and validation.
        If there is a validation set, it'll be 80, 10, and 10.

        Args:
            Folder_path (str): Folder's dataset for distribution

        Returns:
            None
        """
        # * General parameters

        Asterisks: int = 50
        Train_split: float = 0.8
        Test_split: float = 0.1
        Validation_split: float = 0.1

        #Name_dir = os.path.dirname(Folder)
        #Name_base = os.path.basename(Folder)
        #New_Folder_name = Folder_path + '_Split'

        # *
        New_Folder_name = '{}_Split'.format(self.__Folder)

        # *
        print("*" * Asterisks)
        print('New folder name: {}'.format(New_Folder_name))
        print("*" * Asterisks)
    
        splitfolders.ratio(self.__Folder, output = New_Folder_name, seed = 22, ratio = (Train_split, Test_split, Validation_split)) 

        return New_Folder_name

    def split_folders_train_test(self) -> str:

        """
        Create a new folder with the folders of the class problem and its distribution of training and test.
        The split is 80 and 20.

        Args:
            Folder_path (str): Folder's dataset for distribution

        Returns:
            None
        """
        # * General parameters

        Asterisks: int = 50
        Train_split: float = 0.8
        Test_split: float = 0.2

        #Name_dir = os.path.dirname(Folder)
        #Name_base = os.path.basename(Folder)
        #New_Folder_name = Folder_path + '_Split'

        # *
        New_Folder_name = '{}_Split'.format(self.__Folder)

        # *
        print("*" * Asterisks)
        print('New folder name: {}'.format(New_Folder_name))
        print("*" * Asterisks)
    
        splitfolders.ratio(self.__Folder, output = New_Folder_name, seed = 22, ratio = (Train_split, Test_split)) 

        return New_Folder_name
        
# ? .
class DCM_format(Utilities):

    def __init__(self, **kwargs:string) -> None:
        
        # * This algorithm outputs crop values for images based on the coordinates of the CSV file.

        # * Instance attributes folders
        self.__Folder = kwargs.get('folder', None)
        self.__Folder_all = kwargs.get('allfolder', None)
        self.__Folder_patches = kwargs.get('patchesfolder', None)
        self.__Folder_resize = kwargs.get('resizefolder', None)
        self.__Folder_resize_normalize = kwargs.get('normalizefolder', None)

        # * Instance attributes labels
        self.__Severity = kwargs.get('Severity', None)
        self.__Stage = kwargs.get('Phase', None)

        # * Folder attribute (ValueError, TypeError)
        if self.__Folder == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self.Folder, str):
            raise TypeError("Folder must be a string") #! Alert

        # * Folder destination where all the new images will be stored (ValueError, TypeError)
        if self.__Folder_all == None:
            raise ValueError("Destination folder does not exist") #! Alert
        if not isinstance(self.__Folder_all, str):
            raise TypeError("Destination folder must be a string") #! Alert

        # * Folder normal to stored images without preprocessing from CBIS-DDSM (ValueError, TypeError)
        if self.__Folder_patches == None:
            raise ValueError("Normal folder does not exist") #! Alert
        if not isinstance(self.__Folder_patches, str):
            raise TypeError("Normal folder must be a string") #! Alert

        # * Folder normal to stored resize images from CBIS-DDSM (ValueError, TypeError)
        if self.__Folder_resize == None:
            raise ValueError("Resize folder does not exist") #! Alert
        if not isinstance(self.__Folder_resize, str):
            raise TypeError("Resize folder must be a string") #! Alert

        # * Folder normal to stored resize normalize images from CBIS-DDSM (ValueError, TypeError)
        if self.__Folder_resize_normalize == None:
            raise ValueError("Normalize resize folder images does not exist") #! Alert
        if not isinstance(self.__Folder_resize_normalize, str):
            raise TypeError("Normalize resize folder must be a string") #! Alert

        # * Severity label (ValueError, TypeError)
        if self.__Severity == None:
            raise ValueError("Severity does not exist") #! Alert
        if not isinstance(self.__Severity, str):
            raise TypeError("Severity must be a string") #! Alert

        # * Phase label (ValueError, TypeError)
        if self.__Stage == None:
            raise ValueError("Phase images does not exist") #! Alert
        if not isinstance(self.__Stage, str):
            raise TypeError("Phase must be a string") #! Alert

    def __repr__(self) -> str:

            kwargs_info = "Folder: {} , Folder_all: {}, Folder_normal: {}, Folder_resize: {}, Folder_resize_normalize: {}, Severity: {}, Phase: {}".format( self.__Folder, 
                                                                                                                                                            self.__Folder_all, self.__Folder_patches,
                                                                                                                                                            self.__Folder_resize, self.__Folder_resize_normalize, 
                                                                                                                                                            self.__Severity, self.__Stage )
            return kwargs_info

    def __str__(self) -> str:

            Descripcion_class = ""
            
            return Descripcion_class

    # * Folder attribute
    @property
    def Folder_property(self):
        return self.__Folder

    @Folder_property.setter
    def Folder_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Folder must be a string") #! Alert
        self.__Folder = New_value
    
    @Folder_property.deleter
    def Folder_property(self):
        print("Deleting folder...")
        del self.__Folder

    # * Folder all images attribute
    @property
    def Folder_all_property(self):
        return self.__Folder_all

    @Folder_all_property.setter
    def Folder_all_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Folder all must be a string") #! Alert
        self.__Folder_all = New_value
    
    @Folder_all_property.deleter
    def Folder_all_property(self):
        print("Deleting all folder...")
        del self.__Folder_all

    # * Folder patches images attribute
    @property
    def Folder_patches_property(self):
        return self.__Folder_patches

    @Folder_patches_property.setter
    def Folder_patches_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Folder patches must be a string") #! Alert
        self.__Folder_patches = New_value
    
    @Folder_patches_property.deleter
    def Folder_patches_property(self):
        print("Deleting patches folder...")
        del self.__Folder_patches

    # * Folder resize images attribute
    @property
    def Folder_resize_property(self):
        return self.__Folder_resize

    @Folder_resize_property.setter
    def Folder_resize_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Folder resize must be a string") #! Alert
        self.__Folder_resize = New_value
    
    @Folder_resize_property.deleter
    def Folder_resize_property(self):
        print("Deleting resize folder...")
        del self.__Folder_resize

    # * Folder resize normalize images attribute
    @property
    def Folder_resize_normalize_property(self):
        return self.__Folder_resize_normalize

    @Folder_resize_normalize_property.setter
    def Folder_resize_normalize_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Folder resize normalize must be a string") #! Alert
        self.__Folder_resize_normalize = New_value
    
    @Folder_resize_normalize_property.deleter
    def Folder_resize_normalize_property(self):
        print("Deleting resize normalize folder...")
        del self.__Folder_resize_normalize

    # * Severity attribute
    @property
    def Severity_property(self):
        return self.__Severity

    @Severity_property.setter
    def Severity_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Severity must be a string") #! Alert
        self.__Severity = New_value
    
    @Severity_property.deleter
    def Severity_property(self):
        print("Deleting severity...")
        del self.__Severity
    
    # * Stage
    @property
    def Stage_property(self):
        return self.__Stage

    @Stage_property.setter
    def Stage_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Stage must be a string") #! Alert
        self.__Stage = New_value
    
    @Stage_property.deleter
    def Stage_property(self):
        print("Deleting stage...")
        del self.__Stage

    @Utilities.timer_func
    def DCM_change_format(self) -> None:
        """
        Printing amount of images with data augmentation

        Args:
            Folder_path (str): Folder's dataset for distribution

        Returns:
            None
        """

        # * Format DCM and PNG variables
        DCM = ".dcm"
        PNG = ".png"

        # * Initial file
        File = 0

        # * Standard parameters for resize
        X_size_resize = 224
        Y_size_resize = 224

        # * General lists and string DCM
        DCM_files = []
        DCM_files_sizes = []
        DCM_Filenames = []

        # * Interpolation that is used
        Interpolation = cv2.INTER_CUBIC

        # * Shape for the resize
        Shape_resize = (X_size_resize, Y_size_resize)

        # * Read images from folder
        Files_total = os.listdir(self.__Folder)

        # * Sorted files and multiply them
        Files_total_ = Files_total * 2
        Files_total_ = sorted(Files_total_)

        # * Search for each dir and file inside the folder given
        for Root, Dirs, Files in os.walk(self.__Folder, True):
            print("root:%s"% Root)
            print("dirs:%s"% Dirs)
            print("files:%s"% Files)
            print("-------------------------------")

        for Root, Dirs, Files in os.walk(self.__Folder):
            for x in Files:
                if x.endswith(DCM):
                    DCM_files.append(os.path.join(Root, x))

        # * Sorted DCM files
        DCM_files = sorted(DCM_files)
        
        # * Get the size of each dcm file
        for i in range(len(DCM_files)):
            DCM_files_sizes.append(os.path.getsize(DCM_files[i]))

        # * put it together in a dataframe
        DCM_dataframe_files = pd.DataFrame({'Path':DCM_files, 'Size':DCM_files_sizes, 'Filename':Files_total_}) 
        print(DCM_dataframe_files)

        Total_DCM_files = len(DCM_files_sizes)

        # * Search inside each folder to get the archive which has the less size.
        for i in range(0, Total_DCM_files, 2):

            print(DCM_files_sizes[i], '----', DCM_files_sizes[i + 1])

            if DCM_files_sizes[i] > DCM_files_sizes[i + 1]:
                DCM_dataframe_files.drop([i], axis = 0, inplace = True)
            else:
                DCM_dataframe_files.drop([i + 1], axis = 0, inplace = True)

        # * Several prints
        print(len(DCM_files))
        print(len(DCM_files_sizes))
        print(len(Files_total_))
        print(DCM_dataframe_files)

        # * Get the columns of DCM filenames
        DCM_filenames = DCM_dataframe_files.iloc[:, 0].values
        Total_DCM_filenames = DCM_dataframe_files.iloc[:, 2].values

        # * Write the dataframe in a folder
        #DCM_dataframe_name = 'DCM_' + 'Format_' + str(self.Severity) + '_' + str(self.Phase) + '.csv'
        DCM_dataframe_name = 'DCM_Format_{}_{}.csv'.format(str(self.__Severity), str(self.__Stage))
        DCM_dataframe_folder = os.path.join(self.__Folder_all, DCM_dataframe_name)
        DCM_dataframe_files.to_csv(DCM_dataframe_folder)

        # * Convert each image from DCM format to PNG format
        for File in range(len(DCM_dataframe_files)):

            # * Read DCM format using pydicom
            DCM_read_pydicom_file = pydicom.dcmread(DCM_Filenames[File])
            
            # * Convert to float type
            DCM_image = DCM_read_pydicom_file.pixel_array.astype(float)

            # * Rescaled and covert to float64
            DCM_image_rescaled = (np.maximum(DCM_image, 0) / DCM_image.max()) * 255.0
            DCM_image_rescaled_float64 = np.float64(DCM_image_rescaled)

            # * Get a new images to the normalize(zeros)
            DCM_black_image = np.zeros((X_size_resize, Y_size_resize))

            # * Use the resize function
            DCM_image_resize = cv2.resize(DCM_image_rescaled_float64, Shape_resize, interpolation = Interpolation)

            # * Use the normalize function with the resize images
            DCM_image_normalize = cv2.normalize(DCM_image_resize, DCM_black_image, 0, 255, cv2.NORM_MINMAX)

            # * Get each image and convert them
            DCM_file = Total_DCM_filenames[File]
            DCM_name_file = '{}{}'.format(str(DCM_file), str(PNG))
            #DCM_name_file = str(DCM_file) + '.png'

            # * Save each transformation in different folders
            DCM_folder = os.path.join(self.__Folder_patches, DCM_name_file)
            DCM_folder_resize = os.path.join(self.__Folder_resize, DCM_name_file)
            DCM_folder_normalize = os.path.join(self.__Folder_resize_normalize, DCM_name_file)

            cv2.imwrite(DCM_folder, DCM_image_rescaled_float64)
            cv2.imwrite(DCM_folder_resize, DCM_image_resize)
            cv2.imwrite(DCM_folder_normalize, DCM_image_normalize)

            # * Print for comparison
            print('Images: ', DCM_Filenames[File], '------', Total_DCM_filenames[File])    

# ?

class FigureAdjust(Utilities):
  
  def __init__(self, **kwargs) -> None:

    # *
    self._Folder_path = kwargs.get('folder', None)
    self._Title = kwargs.get('title', None)

    # * 
    self._Show_image = kwargs.get('SI', False)
    self._Save_figure = kwargs.get('SF', False)

    # *
    self._Num_classes = kwargs.get('classes', None)

    # *
    self._X_figure_size = 12
    self._Y_figure_size = 12

    # * General parameters
    self._Font_size_title = self._X_figure_size * 1.2
    self._Font_size_general = self._X_figure_size * 0.8
    self._Font_size_ticks = (self._X_figure_size * self._Y_figure_size) * 0.05

    # * 
    #self.Annot_kws = kwargs.get('annot_kws', None)
    #self.Font = kwargs.get('font', None)
  
  def __repr__(self) -> str:

        kwargs_info = ''

        return kwargs_info

  def __str__(self) -> str:

        Descripcion_class = ''
        
        return Descripcion_class

  # * Folder_path attribute
  @property
  def Folder_path_property(self):
      return self._Folder_path

  @Folder_path_property.setter
  def Folder_path_property(self, New_value):
      self._Folder_path = New_value
  
  @Folder_path_property.deleter
  def Folder_path_property(self):
      print("Deleting Folder_path...")
      del self._Folder_path

  # * Title attribute
  @property
  def Title_property(self):
      return self._Title

  @Title_property.setter
  def Title_property(self, New_value):
      self._Title = New_value
  
  @Title_property.deleter
  def Title_property(self):
      print("Deleting Title...")
      del self._Title

  # * Show_image attribute
  @property
  def Show_image_property(self):
      return self._Show_image

  @Show_image_property.setter
  def Show_image_property(self, New_value):
      self._Show_image = New_value
  
  @Show_image_property.deleter
  def Show_image_property(self):
      print("Deleting Show_image...")
      del self._Show_image

  # * Save_figure attribute
  @property
  def Save_figure_property(self):
      return self._Save_figure

  @Save_figure_property.setter
  def Save_figure_property(self, New_value):
      self._Save_figure = New_value
  
  @Save_figure_property.deleter
  def Save_figure_property(self):
      print("Deleting Save_figure...")
      del self._Save_figure

  # * Num_classes attribute
  @property
  def Num_classes_property(self):
      return self._Num_classes

  @Num_classes_property.setter
  def Num_classes_property(self, New_value):
      self._Num_classes = New_value
  
  @Num_classes_property.deleter
  def Num_classes_property(self):
      print("Deleting Num_classes...")
      del self._Num_classes

  # ? Decorator
  @staticmethod
  def show_figure(Show_image: bool = False) -> None:

    if(Show_image == True):
      plt.show()
    
    else: 
      pass

  # ? Decorator
  @staticmethod
  def save_figure(Save_figure: bool, Title: int, Func_: str, Folder: str) -> None:

      if(Save_figure == True):
        
        Figure_name = 'Figure_{}_{}.png'.format(Title, Func_)
        Figure_folder = os.path.join(Folder, Figure_name)
        plt.savefig(Figure_folder)

      else:
        pass
    
# ?
class BarChart(FigureAdjust):
  """
  _summary_

  _extended_summary_
  """
  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)

    # *
    self._CSV_path = kwargs.get('csv', None)

    # *
    self._Plot_x_label = kwargs.get('label', None)
    self._Plot_column = kwargs.get('column', None)
    self._Plot_reverse = kwargs.get('reverse', None)

    # * Read dataframe csv
    self._Dataframe = pd.read_csv(self.CSV_path)

    # *
    self._Colors = ('gray', 'red', 'blue', 'green', 'cyan', 'magenta', 'indigo', 'azure', 'tan', 'purple')
    
    # * General lists
    self._X_fast_list_values = []
    self._X_slow_list_values = []

    self._Y_fast_list_values = []
    self._Y_slow_list_values = []

    self._X_fastest_list_value = []
    self._Y_fastest_list_value = []

    self._X_slowest_list_value = []
    self._Y_slowest_list_value = []

    # * Chosing label
    if self._Num_classes == 2:
      self._Label_class_name = 'Biclass'
    elif self._Num_classes > 2:
      self._Label_class_name = 'Multiclass'

  # * CSV_path attribute
  @property
  def CSV_path_property(self):
      return self._CSV_path

  @CSV_path_property.setter
  def CSV_path_property(self, New_value):
      self._CSV_path = New_value
  
  @CSV_path_property.deleter
  def CSV_path_property(self):
      print("Deleting CSV_path...")
      del self._CSV_path

  # * Plot_x_label attribute
  @property
  def Plot_x_label_property(self):
      return self._Plot_x_label

  @Plot_x_label_property.setter
  def Plot_x_label_property(self, New_value):
      self._Plot_x_label = New_value
  
  @Plot_x_label_property.deleter
  def Plot_x_label_property(self):
      print("Deleting Plot_x_label...")
      del self._Plot_x_label

  # * Plot_column attribute
  @property
  def Plot_column_property(self):
      return self._Plot_column

  @Plot_column_property.setter
  def Plot_column_property(self, New_value):
      self._Plot_column = New_value
  
  @Plot_column_property.deleter
  def Plot_column_property(self):
      print("Deleting Plot_column...")
      del self._Plot_column

  # * Plot_reverse attribute
  @property
  def Plot_reverse_property(self):
      return self._Plot_reverse

  @Plot_reverse_property.setter
  def Plot_reverse_property(self, New_value):
      self._Plot_reverse = New_value
  
  @Plot_reverse_property.deleter
  def Plot_reverse_property(self):
      print("Deleting Plot_reverse...")
      del self._Plot_reverse

  # * Name attribute
  @property
  def Name_property(self):
      return self._Name

  @Name_property.setter
  def Name_property(self, New_value):
      self._Name = New_value
  
  @Name_property.deleter
  def Name_property(self):
      print("Deleting Name...")
      del self._Name

  
  @Utilities.timer_func
  def barchart_horizontal(self) -> None:
    """
	  Show CSV's barchar of all models

    Parameters:
    argument1 (folder): CSV that will be used.
    argument2 (str): Title name.
    argument3 (str): Xlabel name.
    argument1 (dataframe): Dataframe that will be used.
    argument2 (bool): if the value is false, higher values mean better, if the value is false higher values mean worse.
    argument3 (folder): Folder to save the images.
    argument3 (int): What kind of problem the function will classify

    Returns:
	  void
   	"""

    # *
    Horizontal = "horizontal"

    # Initialize the lists for X and Y
    #data = pd.read_csv("D:\MIAS\MIAS VS\DataCSV\DataFrame_Binary_MIAS_Data.csv")

    # * Get X and Y values
    X = list(self._Dataframe.iloc[:, 1])
    Y = list(self._Dataframe.iloc[:, self._Plot_column])

    plt.figure(figsize = (self._X_figure_size, self._Y_figure_size))

    # * Reverse is a bool variable with the postion of the plot
    if self._Plot_reverse == True:

        for Index, (i, k) in enumerate(zip(X, Y)):
            if k < np.mean(Y):
                self._X_fast_list_values.append(i)
                self._Y_fast_list_values.append(k)
            elif k >= np.mean(Y):
                self._X_slow_list_values.append(i)
                self._Y_slow_list_values.append(k)

        for Index, (i, k) in enumerate(zip(self._X_fast_list_values, self._Y_fast_list_values)):
            if k == np.min(self._Y_fast_list_values):
                self._X_fastest_list_value.append(i)
                self._Y_fastest_list_value.append(k)
                #print(X_fastest_list_value)
                #print(Y_fastest_list_value)

        for Index, (i, k) in enumerate(zip(self._X_slow_list_values, self._Y_slow_list_values)):
            if k == np.max(self._Y_slow_list_values):
                self._X_slowest_list_value.append(i)
                self._Y_slowest_list_value.append(k)

    else:

        for Index, (i, k) in enumerate(zip(X, Y)):
            if k < np.mean(Y):
                self._X_slow_list_values.append(i)
                self._Y_slow_list_values.append(k)
            elif k >= np.mean(Y):
                self._X_fast_list_values.append(i)
                self._Y_fast_list_values.append(k)

        for Index, (i, k) in enumerate(zip(self._X_fast_list_values, self._Y_fast_list_values)):
            if k == np.max(self._Y_fast_list_values):
                self._X_fastest_list_value.append(i)
                self._Y_fastest_list_value.append(k)
                #print(XFastest)
                #print(YFastest)

        for Index, (i, k) in enumerate(zip(self._X_slow_list_values, self._Y_slow_list_values)):
            if k == np.min(self._Y_slow_list_values):
                self._X_slowest_list_value.append(i)
                self._Y_slowest_list_value.append(k)

    # * Plot the data using bar() method
    plt.bar(self._X_slow_list_values, self._Y_slow_list_values, label = "Bad", color = 'gray')
    plt.bar(self._X_slowest_list_value, self._Y_slowest_list_value, label = "Worse", color = 'black')
    plt.bar(self._X_fast_list_values, self._Y_fast_list_values, label = "Better", color = 'lightcoral')
    plt.bar(self._X_fastest_list_value, self._Y_fastest_list_value, label = "Best", color = 'red')

    # *
    for Index, value in enumerate(self._Y_slowest_list_value):
        plt.text(0, len(Y) + 3, 'Worse value: {} -------> {}'.format(str(value), str(self._X_slowest_list_value[0])), fontweight = 'bold', fontsize = self._Font_size_general + 1)

    # *
    for Index, value in enumerate(self._Y_fastest_list_value):
        plt.text(0, len(Y) + 4, 'Best value: {} -------> {}'.format(str(value), str(self._X_fastest_list_value[0])), fontweight = 'bold', fontsize = self._Font_size_general + 1)

    plt.legend(fontsize = self._Font_size_general)

    plt.title(self._Title, fontsize = self._Font_size_title)
    plt.xlabel(self._Plot_x_label, fontsize = self._Font_size_general)
    plt.xticks(fontsize = self._Font_size_ticks)
    plt.ylabel("Models", fontsize = self._Font_size_general)
    plt.yticks(fontsize = self._Font_size_ticks)
    plt.grid(color = self._Colors[0], linestyle = '-', linewidth = 0.2)

    # *
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()

    # *
    for i, value in enumerate(self._Y_slow_list_values):
        plt.text(xmax + (0.05 * xmax), i, "{:.8f}".format(value), ha = 'center', fontsize = self._Font_size_ticks, color = 'black')

        Next_value = i

    Next_value = Next_value + 1

    for i, value in enumerate(self._Y_fast_list_values):
        plt.text(xmax + (0.05 * xmax), Next_value + i, "{:.8f}".format(value), ha = 'center', fontsize = self._Font_size_ticks, color = 'black')

    #plt.savefig(Graph_name_folder)

    self.save_figure(self._Save_figure, self._Title, Horizontal, self._Folder_path)
    self.show_figure(self._Show_image)

  @Utilities.timer_func
  def barchart_vertical(self) -> None:  

    """
	  Show CSV's barchar of all models

    Parameters:
    argument1 (folder): CSV that will be used.
    argument2 (str): Title name.
    argument3 (str): Xlabel name.
    argument1 (dataframe): Dataframe that will be used.
    argument2 (bool): if the value is false, higher values mean better, if the value is false higher values mean worse.
    argument3 (folder): Folder to save the images.
    argument3 (int): What kind of problem the function will classify

    Returns:
	  void
   	"""

    # *
    Vertical = "Vertical"

    # Initialize the lists for X and Y
    #data = pd.read_csv("D:\MIAS\MIAS VS\DataCSV\DataFrame_Binary_MIAS_Data.csv")

    # * Get X and Y values
    X = list(self._Dataframe.iloc[:, 1])
    Y = list(self._Dataframe.iloc[:, self._Plot_column])

    plt.figure(figsize = (self._X_figure_size, self._Y_figure_size))

    # * Reverse is a bool variable with the postion of the plot
    if self._Plot_reverse == True:

        for Index, (i, k) in enumerate(zip(X, Y)):
            if k < np.mean(Y):
                self._X_fast_list_values.append(i)
                self._Y_fast_list_values.append(k)
            elif k >= np.mean(Y):
                self._X_slow_list_values.append(i)
                self._Y_slow_list_values.append(k)

        for Index, (i, k) in enumerate(zip(self._X_fast_list_values, self._Y_fast_list_values)):
            if k == np.min(self.Y_fast_list_values):
                self._X_fastest_list_value.append(i)
                self._Y_fastest_list_value.append(k)
                #print(X_fastest_list_value)
                #print(Y_fastest_list_value)

        for Index, (i, k) in enumerate(zip(self._X_slow_list_values, self._Y_slow_list_values)):
            if k == np.max(self._Y_slow_list_values):
                self._X_slowest_list_value.append(i)
                self._Y_slowest_list_value.append(k)

    else:

        for Index, (i, k) in enumerate(zip(X, Y)):
            if k < np.mean(Y):
                self._X_slow_list_values.append(i)
                self._Y_slow_list_values.append(k)
            elif k >= np.mean(Y):
                self._X_fast_list_values.append(i)
                self._Y_fast_list_values.append(k)

        for Index, (i, k) in enumerate(zip(self._X_fast_list_values, self._Y_fast_list_values)):
            if k == np.max(self._Y_fast_list_values):
                self._X_fastest_list_value.append(i)
                self._Y_fastest_list_value.append(k)
                #print(XFastest)
                #print(YFastest)

        for Index, (i, k) in enumerate(zip(self._X_slow_list_values, self._Y_slow_list_values)):
            if k == np.min(self._Y_slow_list_values):
                self._X_slowest_list_value.append(i)
                self._Y_slowest_list_value.append(k)

    # * Plot the data using bar() method
    plt.bar(self._X_slow_list_values, self._Y_slow_list_values, label = "Bad", color = 'gray')
    plt.bar(self._X_slowest_list_value, self._Y_slowest_list_value, label = "Worse", color = 'black')
    plt.bar(self._X_fast_list_values, self._Y_fast_list_values, label = "Better", color = 'lightcoral')
    plt.bar(self._X_fastest_list_value, self._Y_fastest_list_value, label = "Best", color = 'red')

    # *
    for Index, value in enumerate(self._Y_slowest_list_value):
        plt.text(0, len(Y) + 3, 'Worse value: {} -------> {}'.format(str(value), str(self._X_slowest_list_value[0])), fontweight = 'bold', fontsize = self._Font_size_general + 1)

    # *
    for Index, value in enumerate(self._Y_fastest_list_value):
        plt.text(0, len(Y) + 4, 'Best value: {} -------> {}'.format(str(value), str(self._X_fastest_list_value[0])), fontweight = 'bold', fontsize = self._Font_size_general + 1)

    plt.legend(fontsize = self._Font_size_general)

    plt.title(self._Title, fontsize = self._Font_size_title)
    plt.xlabel(self._Plot_x_label, fontsize = self._Font_size_general)
    plt.xticks(fontsize = self._Font_size_ticks)
    plt.ylabel("Models", fontsize = self._Font_size_general)
    plt.yticks(fontsize = self._Font_size_ticks)
    plt.grid(color = self._Colors[0], linestyle = '-', linewidth = 0.2)

    # *
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()

    # *
    for i, value in enumerate(self._Y_slow_list_values):
        plt.text(xmax + (0.05 * xmax), i, "{:.8f}".format(value), ha = 'center', fontsize = self._Font_size_ticks, color = 'black')

        Next_value = i

    Next_value = Next_value + 1

    for i, value in enumerate(self._Y_fast_list_values):
        plt.text(xmax + (0.05 * xmax), Next_value + i, "{:.8f}".format(value), ha = 'center', fontsize = self._Font_size_ticks, color = 'black')

    #plt.savefig(Graph_name_folder)

    self.save_figure(self._Save_figure, self._Title, Vertical, self._Folder_path)
    self.show_figure(self._Show_image)

# ? Create class folders

class FigurePlot(FigureAdjust):
  
  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)

    # * 
    self._Labels = kwargs.get('labels', None)

    # * 
    self._CM_dataframe = kwargs.get('CMdf', None)
    self._History_dataframe = kwargs.get('Hdf', None)
    self._ROC_dataframe = kwargs.get('ROCdf', None)

    # *
    self._X_size_figure_subplot = 2
    self._Y_size_figure_subplot = 2

    # *
    self._Confusion_matrix_dataframe = pd.read_csv(self._CM_dataframe)
    self._History_data_dataframe = pd.read_csv(self._History_dataframe)
    
    self._Roc_curve_dataframes = []
    for Dataframe in self._ROC_dataframe:
      self.Roc_curve_dataframes.append(pd.read_csv(Dataframe))

    # *
    self._Accuracy = self._History_data_dataframe.accuracy.to_list()
    self._Loss = self._History_data_dataframe.loss.to_list()
    self._Val_accuracy = self._History_data_dataframe.val_accuracy.to_list()
    self._Val_loss = self._History_data_dataframe.val_loss.to_list()

    self._FPRs = []
    self._TPRs = []
    for i in range(len(self._Roc_curve_dataframes)):
      self._FPRs.append(self._Roc_curve_dataframes[i].FPR.to_list())
      self._TPRs.append(self._Roc_curve_dataframes[i].TPR.to_list())

  # * CSV_path attribute
  @property
  def CSV_path_property(self):
      return self._CSV_path

  @CSV_path_property.setter
  def CSV_path_property(self, New_value):
      self._CSV_path = New_value
  
  @CSV_path_property.deleter
  def CSV_path_property(self):
      print("Deleting CSV_path...")
      del self._CSV_path

  # * Roc_curve_dataframe attribute
  @property
  def Roc_curve_dataframe_property(self):
      return self._Roc_curve_dataframe

  @Roc_curve_dataframe_property.setter
  def Roc_curve_dataframe_property(self, New_value):
      self._Roc_curve_dataframe = New_value
  
  @Roc_curve_dataframe_property.deleter
  def Roc_curve_dataframe_property(self):
      print("Deleting Roc_curve_dataframe...")
      del self._Roc_curve_dataframe

  @Utilities.timer_func
  def figure_plot_four(self) -> None: 

    # *
    Four_plot = 'Four_plot'

    # * Figure's size
    plt.figure(figsize = (self._X_figure_size, self._Y_figure_size))
    plt.suptitle(self._Title, fontsize = self._Font_size_title)
    plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 4)

    # * Confusion matrix heatmap
    sns.set(font_scale = self._Font_size_general)

    # *
    ax = sns.heatmap(self._Confusion_matrix_dataframe, annot = True, fmt = 'd', annot_kws = {"size": self._Font_size_general})
    #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')

    # * Subplot training accuracy
    plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 1)
    plt.plot(self._Accuracy, label = 'Training Accuracy')
    plt.plot(self._Val_accuracy, label = 'Validation Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc = 'lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')

    # * Subplot training loss
    plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 2)
    plt.plot(self._Loss, label = 'Training Loss')
    plt.plot(self._Val_loss, label = 'Validation Loss')
    plt.ylim([0, 2.0])
    plt.legend(loc = 'upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')

    # * FPR and TPR values for the ROC curve
    Auc = auc(self._FPRs[0], self._TPRs[0])

    # * Subplot ROC curve
    plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 3)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(self._FPRs[0], self._TPRs[0], label = 'Test' + '(area = {:.4f})'.format(Auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc = 'lower right')
    
    self.save_figure(self._Save_figure, self._Title, Four_plot, self._Folder_path)
    self.show_figure(self._Show_image)

  @Utilities.timer_func
  def figure_plot_four_multiclass(self) -> None: 
    
    # * Colors for ROC curves
    Colors = ['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan']

    # *
    Four_plot = 'Four_plot'
    Roc_auc = dict()


    # * Figure's size
    plt.figure(figsize = (self._X_figure_size, self._Y_figure_size))
    plt.suptitle(self._Title, fontsize = self._Font_size_title)
    plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 4)

    # * Confusion matrix heatmap
    sns.set(font_scale = self._Font_size_general)

    # *
    ax = sns.heatmap(self._Confusion_matrix_dataframe, annot = True, fmt = 'd', annot_kws = {"size": self._Font_size_general})
    #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')

    # * Subplot training accuracy
    plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 1)
    plt.plot(self._Accuracy, label = 'Training Accuracy')
    plt.plot(self._Val_accuracy, label = 'Validation Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc = 'lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')

    # * Subplot training loss
    plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 2)
    plt.plot(self._Loss, label = 'Training Loss')
    plt.plot(self._Val_loss, label = 'Validation Loss')
    plt.ylim([0, 2.0])
    plt.legend(loc = 'upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')

    # * FPR and TPR values for the ROC curve
    for i in range(len(self.Roc_curve_dataframes)):
      Roc_auc[i] = auc(self._FPRs[i], self._TPRs[i])

    # * Plot ROC curve
    plt.subplot(self._X_size_figure_subplot, self._Y_size_figure_subplot, 3)
    plt.plot([0, 1], [0, 1], 'k--')

    for i in range(len(self.Roc_curve_dataframes)):
      plt.plot(self._FPRs[i], self._TPRs[i], color = Colors[i], label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(self._Labels[i], Roc_auc[i]))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc = 'lower right')
    
    self.save_figure(self._Save_figure, self._Title, Four_plot, self._Folder_path)
    self.show_figure(self._Show_image)

  @Utilities.timer_func
  def figure_plot_CM(self) -> None:
    
    # *
    CM_plot = 'CM_plot'

    # *
    Confusion_matrix_dataframe = pd.read_csv(self._CM_dataframe)

    # * Figure's size
    plt.figure(figsize = (self._X_figure_size / 2, self._Y_figure_size / 2))
    plt.title('Confusion Matrix with {}'.format(self._Title))

    # * Confusion matrix heatmap
    sns.set(font_scale = self._Font_size_general)

    # *
    ax = sns.heatmap(Confusion_matrix_dataframe, annot = True, fmt = 'd', annot_kws = {"size": self._Font_size_general})
    #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')

    self.save_figure(self._Save_figure, self._Title, CM_plot, self._Folder_path)
    self.show_figure(self._Show_image)

  @Utilities.timer_func
  def figure_plot_acc(self) -> None:

    # *
    ACC_plot = 'ACC_plot'

    # * Figure's size
    plt.figure(figsize = (self._X_figure_size / 2, self._Y_figure_size / 2))
    plt.title('Training and Validation Accuracy with {}'.format(self._Title))

    # * Plot training accuracy
    plt.plot(self._Accuracy, label = 'Training Accuracy')
    plt.plot(self._Val_accuracy, label = 'Validation Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc = 'lower right')
    plt.xlabel('Epoch')

    self.save_figure(self._Save_figure, self._Title, ACC_plot, self._Folder_path)
    self.show_figure(self._Show_image)

  @Utilities.timer_func
  def figure_plot_loss(self) -> None:

    # *
    Loss_plot = 'Loss_plot'

    # * Figure's size
    
    plt.figure(figsize = (self._X_figure_size / 2, self._Y_figure_size / 2))
    plt.title('Training and Validation Loss with {}'.format(self._Title))

    # * Plot training loss
    plt.plot(self._Loss, label = 'Training Loss')
    plt.plot(self._Val_loss, label = 'Validation Loss')
    plt.ylim([0, 2.0])
    plt.legend(loc = 'upper right')
    plt.xlabel('Epoch')

    self.save_figure(self._Save_figure, self._Title, Loss_plot, self._Folder_path)
    self.show_figure(self._Show_image)

  @Utilities.timer_func
  def figure_plot_ROC_curve(self) -> None:
    
    # *
    ROC_plot = 'ROC_plot'

    # * Figure's size
    plt.figure(figsize = (self._X_figure_size / 2, self._Y_figure_size / 2))
    plt.title('ROC curve Loss with {}'.format(self.Title))

    # * FPR and TPR values for the ROC curve
    AUC = auc(self._FPRs[0], self._TPRs[0])

    # * Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(self._FPRs[0], self._TPRs[0], label = 'Test' + '(area = {:.4f})'.format(AUC))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc = 'lower right')

    self.save_figure(self._Save_figure, self._Title, ROC_plot, self._Folder_path)
    self.show_figure(self._Show_image)

  def figure_plot_ROC_curve_multiclass(self) -> None:

    # * Colors for ROC curves
    Colors = ['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan']

    # *
    ROC_plot = 'ROC_plot'
    Roc_auc = dict()

    # * Figure's size
    plt.figure(figsize = (self._X_figure_size / 2, self._Y_figure_size / 2))
    plt.title(self._Title, fontsize = self._Font_size_title)

    # * FPR and TPR values for the ROC curve
    for i in range(len(self.Roc_curve_dataframes)):
      Roc_auc[i] = auc(self._FPRs[i], self._TPRs[i])

    # * Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')

    for i in range(len(self.Roc_curve_dataframes)):
      plt.plot(self._FPRs[i], self._TPRs[i], color = Colors[i], label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(self._Labels[i], Roc_auc[i]))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc = 'lower right')

    self.save_figure(self._Save_figure, self._Title, ROC_plot, self._Folder_path)
    self.show_figure(self._Show_image)