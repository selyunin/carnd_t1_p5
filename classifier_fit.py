#!/usr/bin/env python

'''
Created on Jan 6, 2018
@author: selyunin
'''

import os
import cv2
import glob
import time
import zipfile
import pickle
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cross_validation import train_test_split

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

class ClassifierFit():
    def __init__(self, *args, **kwargs):
        self.params = {}
        self.get_training_data()
        self.get_img_names()
        self.print_img_stats()
        self.split_train_test_data()
        self.print_train_test_stats()
        self.get_params()
        self.compute_features()
        self.scale_features()
        self.fit_classifier()
        self.store_classifier()
        
    def fit_classifier(self):
        print("Fitting data to LinearSVC...")
        
        self.svc = LinearSVC()

        self.svc.fit(self.X_train, self.y_train)
        # Check the score of the SVC
        validation_acc = self.svc.score(self.X_val, self.y_val)
        test_acc = self.svc.score(self.X_test, self.y_test)
        print("Validation Accuracy  = {}".format(validation_acc))
        print("Test Accuracy        = {}".format(test_acc))
        
    def store_classifier(self):
        self.pickle_name = "svc_pickle_{}.p".format(self.params['color_space'])
        svc_data_pickle = {}
        svc_data_pickle["SVC"] = self.svc
        svc_data_pickle["scaler"] = self.scaler
        svc_data_pickle["params"] = self.params
        with open(self.pickle_name, "wb") as p:
            pickle.dump( svc_data_pickle, p )
        
    def get_params(self):
        ### Tweak these parameters and see how the results change.
        self.params['color_space'] = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.params['orient'] = 11  # HOG orientations
        self.params['pix_per_cell'] = 8 # HOG pixels per cell
        self.params['cell_per_block'] = 2 # HOG cells per block
        self.params['hog_channel'] = 'ALL' # Can be 0, 1, 2, or "ALL"
        self.params['spatial_size'] = (16, 16) # Spatial binning dimensions
        self.params['hist_bins'] = 32    # Number of histogram bins
        self.params['spatial_feat'] = True # Spatial features on or off
        self.params['hist_feat'] = True # Histogram features on or off
        self.params['hog_feat'] = True # HOG features on or off
        
    def compute_features(self):
        t_start = time.time()
        print("Starting HOG feature extraction: {} s".format(t_start))
        self.vehicles_training_features = self.extract_features(self.vehicles_train, 
                                       color_space=self.params['color_space'], 
                                       spatial_size=self.params['spatial_size'], 
                                       hist_bins=self.params['hist_bins'], 
                                       orient=self.params['orient'], 
                                       pix_per_cell=self.params['pix_per_cell'], 
                                       cell_per_block=self.params['cell_per_block'], 
                                       hog_channel=self.params['hog_channel'], 
                                       spatial_feat=self.params['spatial_feat'], 
                                       hist_feat=self.params['hist_feat'], 
                                       hog_feat=self.params['hog_feat'])
        
        self.vehicles_validation_features = self.extract_features(self.vehicles_val, 
                                       color_space=self.params['color_space'], 
                                       spatial_size=self.params['spatial_size'], 
                                       hist_bins=self.params['hist_bins'], 
                                       orient=self.params['orient'], 
                                       pix_per_cell=self.params['pix_per_cell'], 
                                       cell_per_block=self.params['cell_per_block'], 
                                       hog_channel=self.params['hog_channel'], 
                                       spatial_feat=self.params['spatial_feat'], 
                                       hist_feat=self.params['hist_feat'], 
                                       hog_feat=self.params['hog_feat'])
        
        self.vehicles_testing_features = self.extract_features(self.vehicles_test, 
                                       color_space=self.params['color_space'], 
                                       spatial_size=self.params['spatial_size'], 
                                       hist_bins=self.params['hist_bins'], 
                                       orient=self.params['orient'], 
                                       pix_per_cell=self.params['pix_per_cell'], 
                                       cell_per_block=self.params['cell_per_block'], 
                                       hog_channel=self.params['hog_channel'], 
                                       spatial_feat=self.params['spatial_feat'], 
                                       hist_feat=self.params['hist_feat'], 
                                       hog_feat=self.params['hog_feat'])
        
        self.nonvehicles_training_features = self.extract_features(self.nonvehicles_train, 
                                       color_space=self.params['color_space'], 
                                       spatial_size=self.params['spatial_size'], 
                                       hist_bins=self.params['hist_bins'], 
                                       orient=self.params['orient'], 
                                       pix_per_cell=self.params['pix_per_cell'], 
                                       cell_per_block=self.params['cell_per_block'], 
                                       hog_channel=self.params['hog_channel'], 
                                       spatial_feat=self.params['spatial_feat'], 
                                       hist_feat=self.params['hist_feat'], 
                                       hog_feat=self.params['hog_feat'])
        
        self.nonvehicles_validation_features = self.extract_features(self.nonvehicles_val, 
                                       color_space=self.params['color_space'], 
                                       spatial_size=self.params['spatial_size'], 
                                       hist_bins=self.params['hist_bins'], 
                                       orient=self.params['orient'], 
                                       pix_per_cell=self.params['pix_per_cell'], 
                                       cell_per_block=self.params['cell_per_block'], 
                                       hog_channel=self.params['hog_channel'], 
                                       spatial_feat=self.params['spatial_feat'], 
                                       hist_feat=self.params['hist_feat'], 
                                       hog_feat=self.params['hog_feat'])
        
        self.nonvehicles_testing_features = self.extract_features(self.nonvehicles_test, 
                                       color_space=self.params['color_space'], 
                                       spatial_size=self.params['spatial_size'], 
                                       hist_bins=self.params['hist_bins'], 
                                       orient=self.params['orient'], 
                                       pix_per_cell=self.params['pix_per_cell'], 
                                       cell_per_block=self.params['cell_per_block'], 
                                       hog_channel=self.params['hog_channel'], 
                                       spatial_feat=self.params['spatial_feat'], 
                                       hist_feat=self.params['hist_feat'], 
                                       hog_feat=self.params['hog_feat'])
        
        t_finish = time.time()
        print("HOG features extracted in {} s".format(t_finish - t_start))
        
    def scale_features(self):
        # Create an array stack of feature vectors
        self.stacked_features = np.vstack((self.vehicles_training_features,
                                      self.vehicles_validation_features,
                                      self.vehicles_testing_features,
                                      self.nonvehicles_training_features,
                                      self.nonvehicles_validation_features,
                                      self.nonvehicles_testing_features)).astype(np.float64)                        
        # Fit a per-column scaler
        self.scaler = StandardScaler().fit(self.stacked_features)
        # Apply the scaler to X
        self.scaled_stacked_features = self.scaler.transform(self.stacked_features)
    
        idx_veh_train    = (0,                   len(self.vehicles_training_features))
        idx_veh_val      = (idx_veh_train[1],    idx_veh_train[1] + len(self.vehicles_validation_features))
        idx_veh_test     = (idx_veh_val[1],      idx_veh_val[1] + len(self.vehicles_testing_features))
        idx_nonveh_train = (idx_veh_test[1],     idx_veh_test[1] + len(self.nonvehicles_training_features))
        idx_nonveh_val   = (idx_nonveh_train[1], idx_nonveh_train[1] + len(self.nonvehicles_validation_features))
        idx_nonveh_test  = (idx_nonveh_val[1],   idx_nonveh_val[1] + len(self.nonvehicles_testing_features))
        
        self.scaled_vehicles_training_features      = self.scaled_stacked_features[idx_veh_train[0]:idx_veh_train[1]]
        self.scaled_vehicles_validation_features    = self.scaled_stacked_features[idx_veh_val[0]:idx_veh_val[1]]
        self.scaled_vehicles_testing_features       = self.scaled_stacked_features[idx_veh_test[0]:idx_veh_test[1]]
        self.scaled_nonvehicles_training_features   = self.scaled_stacked_features[idx_nonveh_train[0]:idx_nonveh_train[1]]
        self.scaled_nonvehicles_validation_features = self.scaled_stacked_features[idx_nonveh_val[0]:idx_nonveh_val[1]]
        self.scaled_nonvehicles_testing_features    = self.scaled_stacked_features[idx_nonveh_test[0]:idx_nonveh_test[1]]
        
        
        self.y_train = np.hstack((np.ones(len(self.vehicles_training_features)), 
                             np.zeros(len(self.nonvehicles_training_features))))
        self.y_val   = np.hstack((np.ones(len(self.vehicles_validation_features)), 
                             np.zeros(len(self.nonvehicles_validation_features))))
        self.y_test  = np.hstack((np.ones(len(self.vehicles_testing_features)), 
                             np.zeros(len(self.nonvehicles_testing_features))))
        
        self.X_train = np.vstack((self.scaled_vehicles_training_features,
                                  self.scaled_nonvehicles_training_features))
        self.X_val   = np.vstack((self.scaled_vehicles_validation_features,
                                  self.scaled_nonvehicles_validation_features))
        self.X_test  = np.vstack((self.scaled_vehicles_testing_features,
                                  self.scaled_nonvehicles_testing_features))
        
        self.X_train_img = np.concatenate((self.vehicles_train,
                                           self.nonvehicles_train))
        self.X_val_img = np.concatenate((self.vehicles_val,
                                         self.nonvehicles_val))
        self.X_test_img = np.concatenate((self.vehicles_test,
                                          self.nonvehicles_test))
        
        self.random_state=int(time.time())
        
        self.X_train,self.y_train = shuffle(self.X_train,self.y_train,random_state=self.random_state)
        self.X_val,self.y_val = shuffle(self.X_val,self.y_val,random_state=self.random_state)
        self.X_test,self.y_test = shuffle(self.X_test,self.y_test,random_state=self.random_state)
        
        self.X_train_img = shuffle(self.X_train_img, random_state=self.random_state)
        self.X_val_img = shuffle(self.X_val_img, random_state=self.random_state)
        self.X_test_img = shuffle(self.X_test_img, random_state=self.random_state)
        
        print("X_train_img: {}\t\tX_train: {}".format(self.X_train_img.shape, self.X_train.shape))
        print("X_val_img: {}\t\tX_val: {}".format(self.X_val_img.shape, self.X_val.shape))
        print("X_test_img: {}\t\tX_test: {}".format(self.X_test_img.shape, self.X_test.shape))
        
    
    def get_training_data(self):
            # Load image training dataset
        self.vehicles_url     = 'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip'
        self.vehicles_arch    = './vehicles.zip'
        self.vehicles_dir     = './vehicles'
        self.vehicles_sub_folders = [f for f in os.listdir(self.vehicles_dir) if '.DS_Store' not in f]
        self.nonvehicles_url  = 'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip'
        self.nonvehicles_arch = './non-vehicles.zip'
        self.nonvehicles_dir  = './non-vehicles'
        self.nonvehicles_sub_folders = [f for f in os.listdir(self.nonvehicles_dir) if '.DS_Store' not in f]
        
        if not os.path.isfile(self.vehicles_arch):
            print("Downloading dataset of vehicles from the web...")
            # download the dataset from the Web
            urllib.request.urlretrieve(self.vehicles_url, 
                                       self.vehicles_arch)
        else:
            print("Dataset of vehicles was downloaded before..")
        
        if not os.path.isdir(self.vehicles_dir):
            print("Extracting the {} archive...".format(self.vehicles_arch))
            # extract images from the zip archive
            zip_ref = zipfile.ZipFile(self.vehicles_arch, 'r') 
            zip_ref.extractall('./')
            zip_ref.close()
            print("Extracting vehicles done...")
            
        if not os.path.isfile(self.nonvehicles_arch):
            print("Downloading dataset of non-vehicles from the web...")
            # download the dataset from the Web
            urllib.request.urlretrieve(self.nonvehicles_url, 
                                       self.nonvehicles_arch)
        else:
            print("Dataset of non-vehicles was downloaded before..")
            
        if not os.path.isdir(self.nonvehicles_dir):
            print("Extracting the {} archive...".format(self.nonvehicles_arch))
            # extract images from the zip archive
            zip_ref = zipfile.ZipFile(self.nonvehicles_arch, 'r') 
            zip_ref.extractall('./')
            zip_ref.close()
            print("Extracting non-vehicles done...")
    
    def get_img_names(self):
        self.vehicles_img_names = []
        for path, subdirs, files in os.walk(self.vehicles_dir):
            for name in files:
                 if '.DS_Store' not in name:
                    self.vehicles_img_names.append(os.path.join(path, name))
        
        self.nonvehicles_img_names = []
        for path, subdirs, files in os.walk(self.nonvehicles_dir):
            for name in files:
                 if '.DS_Store' not in name:
                    self.nonvehicles_img_names.append(os.path.join(path, name))
                    
        #for each vehicle (non-vehicle) sub-folder do a dictionary folder --> img names
        self.vehicles_data_subsets = {}
        for sub_folder in self.vehicles_sub_folders:
            sub_folder_path = os.path.join(self.vehicles_dir, sub_folder)
            self.vehicles_data_subsets[sub_folder] = glob.glob(sub_folder_path + '/' + '*.png')
        
        self.nonvehicles_data_subsets = {}
        for sub_folder in self.nonvehicles_sub_folders:
            sub_folder_path = os.path.join(self.nonvehicles_dir, sub_folder)
            self.nonvehicles_data_subsets[sub_folder] = glob.glob(sub_folder_path + '/' + '*.png')   
            
    def split_train_test_data(self):
        seed = int(time.time())
        
        self.vehicles_train, self.vehicles_val_test = train_test_split(self.vehicles_img_names, 
                                                             train_size=0.65, 
                                                             test_size=0.35,
                                                             random_state=seed)
        self.vehicles_val, self.vehicles_test = train_test_split(self.vehicles_val_test, 
                                                       train_size=0.6, 
                                                       test_size=0.4,
                                                       random_state=seed)
        
        self.nonvehicles_train, self.nonvehicles_val_test = train_test_split(self.nonvehicles_img_names, 
                                                                   train_size=0.65, 
                                                                   test_size=0.35,
                                                                   random_state=seed)
        self.nonvehicles_val, self.nonvehicles_test = train_test_split(self.nonvehicles_val_test, 
                                                             train_size=0.6, 
                                                             test_size=0.4,
                                                             random_state=seed)
    
        # Define a function to return HOG features and visualization
    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, 
                            vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient, 
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), 
                                      transform_sqrt=True, 
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:      
            features = hog(img, orientations=orient, 
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), 
                           transform_sqrt=True, 
                           visualise=vis, feature_vector=feature_vec)
            return features
    
    # Define a function to compute binned color features  
    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel() 
        # Return the feature vector
        return features
    
    # Define a function to compute color histogram features 
    # NEED TO CHANGE bins_range if reading .png files with mpimg!
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features
    
    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, imgs, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9, 
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=True, hist_feat=True, hog_feat=True):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for fname in imgs:
            file_features = []
            # Read in each one by one
            image = cv2.imread(fname)
            # apply color conversion if other than 'RGB'
            if color_space == 'RGB':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
            if spatial_feat == True:
                spatial_features = self.bin_spatial(feature_image, size=spatial_size)
                file_features.append(spatial_features)
            if hist_feat == True:
                # Apply color_hist()
                hist_features = self.color_hist(feature_image, nbins=hist_bins)
                file_features.append(hist_features)
            if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(self.get_hog_features(feature_image[:,:,channel], 
                                            orient, pix_per_cell, cell_per_block, 
                                            vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)        
                else:
                    hog_features = self.get_hog_features(feature_image[:,:,hog_channel], orient, 
                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        # Return list of feature vectors
        return features
    
    def print_train_test_stats(self):    
        print("Vehicles training examples       : {}".format(len(self.vehicles_train)))
        print("Vehicles validation examples     : {}".format(len(self.vehicles_val)))
        print("Vehicles testing examples        : {}".format(len(self.vehicles_test)))
        print("")
        print("Non-Vehicles training examples   : {}".format(len(self.nonvehicles_train)))
        print("Non-Vehicles validation examples : {}".format(len(self.nonvehicles_val)))
        print("Non-Vehicles testing examples    : {}".format(len(self.nonvehicles_test)))            
                    
    def print_img_stats(self):
        print("Number of     vehicle images = {}".format(len(self.vehicles_img_names)))
        print("Number of non-vehicle images = {}".format(len(self.nonvehicles_img_names)))
        print("Image data shape = {}\n\n".format(mpimg.imread(self.vehicles_img_names[1]).shape))
        
        print("Vehicles:")
        for key, val in self.vehicles_data_subsets.items():
            print("Subset: {:20s} #images: {}".format(key, len(val)))
        
        print("\nNon-Vehicles:")
        for key, val in self.nonvehicles_data_subsets.items():
            print("Subset: {:20s} #images: {}".format(key, len(val)))