import os
from glob import glob

# Method to get info to use changedetection dataset as a list with each video as a element of the shape (path, dataset, category)
def get_changeDetection_dirs_and_info(dataset_path, category_to_search = ["ALL"], category_to_ignore = [], video_to_search = ["ALL"], video_to_ignore = [], do_print = False):
    
    videos_info = []
    if os.path.isdir(dataset_path):
        if do_print:
            print (dataset_path)
            print("changedetection videos info:")
        if dataset_path[-1] != '*':
            dataset_path = os.path.join(dataset_path,'*')                                                           # We prepare it for glob.                                                                             
        subfolders_in_folder = sorted(glob(dataset_path))
        for subfolder_path in subfolders_in_folder:                                                                 # We need to process each dataset...
            if os.path.isdir(subfolder_path) and ("ALL" in category_to_search or os.path.basename(subfolder_path) in category_to_search) and not (os.path.basename(subfolder_path) in category_to_ignore):              # If this element is an important folder...
                elements_in_subfolder = sorted(glob(os.path.join(subfolder_path, "*")))                             # Get elements in subfolder...
                for element_path in elements_in_subfolder:                    
                    if os.path.isdir(element_path) and ("ALL" in video_to_search or os.path.basename(element_path) in video_to_search) and not (os.path.basename(subfolder_path) in video_to_ignore):                   # If this element is a folder...
                        category_path = subfolder_path                                                              # We will consider subfolder as a category
                        _, category = os.path.split(category_path)
                        dataset_path = element_path                                                                 # and this element as a dataset.
                        _, dataset = os.path.split(dataset_path)
                        dataset_elements = sorted(glob(os.path.join(dataset_path, "*")))
                        if os.path.join(dataset_path, "input") in dataset_elements:                                 # If there is an input folder
                            dataset_path=os.path.join(dataset_path, "input")                                        # we will get path from input folder.

                        videos_info.append((dataset_path, dataset, category))
                        if do_print:
                            print("("+dataset_path+", "+dataset+", "+category+")")
                    else:
                        if not os.path.isdir(element_path):
                            raise("Error: " + dataset_path + " is not a folder and it should be a dataset.")        # Error.            
            else:
                if not os.path.isdir(subfolder_path):
                    raise("Error: " + subfolder_path + " is not a folder")                                          # Error. There is no folder.
    return videos_info


def obtain_information_about_dataset(dataset_name):

    if (dataset_name == "highway"):
        number_of_training_images = 469 #Number of images to obtain initial gaussian models.
        category ="baseline"
        
    if (dataset_name == "office"):
        number_of_training_images = 569 #Number of images to obtain initial gaussian models.
        category ="baseline"
        
    if (dataset_name == "pedestrians"):
        number_of_training_images = 299 #Number of images to obtain initial gaussian models.
        category ="baseline"
        
    if (dataset_name == "PETS2006"):
        number_of_training_images = 299 #Number of images to obtain initial gaussian models.
        category ="baseline"
        
    if (dataset_name == "canoe"):
        number_of_training_images = 799 #Number of images to obtain initial gaussian models.
        category ="dynamicBackground"
        
    if (dataset_name == "boats"):
        number_of_training_images = 1899 #Number of images to obtain initial gaussian models.
        category ="dynamicBackground"
        
    if (dataset_name == "port_0_17fps"):
        number_of_training_images = 999 #Number of images to obtain initial gaussian models.
        category ="lowFramerate"
        
    if (dataset_name == "overpass"):
        number_of_training_images = 999 #Number of images to obtain initial gaussian models.
        category ="dynamicBackground"
        
    if (dataset_name == "streetCornerAtNight"):
        number_of_training_images = 799 #Number of images to obtain initial gaussian models.
        category ="nightVideos"
        
    if (dataset_name == "tramStation"):
        number_of_training_images = 499 #Number of images to obtain initial gaussian models.
        category = "nightVideos"
        
    if (dataset_name == "blizzard"):
        number_of_training_images = 899 #Number of images to obtain initial gaussian models.
        category ="badWeather"

    if (dataset_name == "fall"):
        number_of_training_images = 999 #Number of images to obtain initial gaussian models.
        category ="dynamicBackground"
        
    if (dataset_name == "wetSnow"):
        number_of_training_images = 499 #Number of images to obtain initial gaussian models.
        category ="badWeather"
        
    if (dataset_name == "skating"):
        number_of_training_images = 799 #Number of images to obtain initial gaussian models.
        category ="badWeather"
        our_prefix = "bin"
        
    if (dataset_name == "snowFall"):
        number_of_training_images = 799 #Number of images to obtain initial gaussian models.
        category ="badWeather"
        
    if (dataset_name == "fountain01"):
        number_of_training_images = 399 #Number of images to obtain initial gaussian models.
        category ="dynamicBackground"
        
    if (dataset_name == "fountain02"):
        number_of_training_images = 499 #Number of images to obtain initial gaussian models.
        category ="dynamicBackground"
     
    return category, number_of_training_images
    
def get_CDNET_number_of_training_images(category, dataset, DATASET_FOLDER = "/usr/share/Data1/Datasets/changeDetection/"):
    temporal_roi_path = os.path.join(DATASET_FOLDER, category, dataset, "temporalROI.txt")
    with open(temporal_roi_path, 'r') as f:
        line = f.readline()
        splitted_line = line.split()
        #print(splitted_line)
        first_value_in_ROI = splitted_line[0]
        #print(first_value_in_ROI)
        n_training_images = (int(first_value_in_ROI)-1)
        #print(n_training_images)
        return n_training_images
        
