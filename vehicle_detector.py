'''
Created on Jan 6, 2018
@author: selyunin
'''

import cv2
import numpy as np
import pickle

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from scipy.ndimage.measurements import label
from vehicle import Vehicle


from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.filters import gaussian_filter

class VehicleDetector():
    def __init__(self, *args, **kwargs):
        self.vehicles = Vehicle()
        self.svc_pickle_file = 'svc_pickle.p'
        self.svc = None
        self.scaler = None
        self.get_svc_classifier()
        print("svc: {}".format(type(self.svc)))
        print("sca: {}".format(type(self.scaler)))
        
        self.xy_window = None
        self.xy_overlap = None
        self.x_start_stop = None
        self.y_top = None
        self.y_start_stop = None
        self.color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (16, 16) # Spatial binning dimensions
        self.hist_bins = 32    # Number of histogram bins
        self.spatial_feat = True # Spatial features on or off
        self.hist_feat = True # Histogram features on or off
        self.hog_feat = True # HOG features on or off
        self.get_window_coordinates()
        self.get_more_classifiers()
#         self.xy_window = None
#         self.xy_window = None
    
    def get_svc_classifier(self):
        with(open(self.svc_pickle_file, 'rb')) as p:
            svc_pickle = pickle.load(p)
            self.svc = svc_pickle['SVC']
            self.scaler = svc_pickle['scaler']
#             vehicles_val = data_pickle["vehicles_val"]
#             vehicles_test = data_pickle["vehicles_test"]
#             nonvehicles_train = data_pickle["nonvehicles_train"]
#             nonvehicles_val = data_pickle["nonvehicles_val"]
#             nonvehicles_test = data_pickle["nonvehicles_test"]


    def get_more_classifiers(self):
        self.classifiers = {}
        self.classifiers_pickle  = [
#             'svc_pickle_HLS.p',
#             'svc_pickle_HSV.p',
            'svc_pickle_YCrCb.p',
            'svc_pickle_YUV.p'
            ]
        for p_file in self.classifiers_pickle:
            self.classifiers[p_file] = {}
            with(open(p_file, 'rb')) as p:
                clf_pickle = pickle.load(p)
                self.classifiers[p_file]['svc'] = clf_pickle['SVC']
                self.classifiers[p_file]['scaler'] = clf_pickle['scaler']
                self.classifiers[p_file]['params'] = clf_pickle['params']
        for key, val in self.classifiers.items():
            print ("clf: {}, params {}".format(key, val))
#             vehicles_val = data_pickle["vehicles_val"]
#             vehicles_test = data_pickle["vehicles_test"]
#             nonvehicles_train = data_pickle["nonvehicles_train"]
#             nonvehicles_val = data_pickle["nonvehicles_val"]
#             nonvehicles_test = data_pickle["nonvehicles_test"]
    
    def process_image(self, img):
        
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        
        box_list = self.get_on_windows_v2(img)
        
        # Add heat to each box in box list
        heat = self.add_heat(heat,box_list)
        
        # Apply threshold to help remove false positives
        heat = self.apply_threshold(heat, 2)
        heat = gaussian_filter(heat, sigma=0.2, order=0)
#         self.vehicles.add_heat(heat)
        self.vehicles.add_heat(heat)
        heat = self.vehicles.get_heat()
        heat = self.apply_threshold(heat,len(self.vehicles.heat))
#         self.vehicles.add_heat(heat)
#         print(np.max(heat))
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
#         heatmap = np.uint8(gaussian_filter(heatmap, sigma=0.25, order=0))
#         myheatmap = np.uint8(np.dstack( (heatmap, heatmap, heatmap) ))
#         print(myheatmap.shape)
        
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        
#         print("labels: {}".format(labels[0].shape))
#         print("cars detected: {}".format(labels[1]))
        
        self.vehicles.add_labels(labels)
#         print("labels: {}".format(labels))
        draw_img = self.draw_labeled_bboxes(np.copy(img), labels)
        
        window_img = self.draw_boxes(np.copy(img), box_list, color=(0, 255, 255), thick=4)                    

#         return window_img
        return draw_img
    
    def get_on_windows_v2(self, image):
        draw_image = np.copy(image)
        on_windows = []
#         all_windows = []
        window_params = list(zip(self.x_start_stop, 
                                 self.y_start_stop, 
                                 self.xy_window, 
                                 self.xy_overlap))
        for el_x_start_stop, el_y_start_stop, el_xy_window, el_xy_overlap in window_params:
            windows = self.slide_window(image, 
                                   x_start_stop = el_x_start_stop, 
                                   y_start_stop = el_y_start_stop, 
                                   xy_window = el_xy_window, 
                                   xy_overlap = el_xy_overlap)
        
#             all_windows.extend([windows])
            for clf in self.classifiers:
                
                curr_on_windows = self.search_windows(image, 
                                           windows=windows, 
                                           clf=self.classifiers[clf]['svc'], 
                                           scaler=self.classifiers[clf]['scaler'], 
                                           color_space=self.classifiers[clf]['params']['color_space'], 
                                           spatial_size=self.classifiers[clf]['params']['spatial_size'], 
                                           hist_bins=self.classifiers[clf]['params']['hist_bins'], 
                                           orient=self.classifiers[clf]['params']['orient'], 
                                           pix_per_cell=self.classifiers[clf]['params']['pix_per_cell'], 
                                           cell_per_block=self.classifiers[clf]['params']['cell_per_block'], 
                                           hog_channel=self.classifiers[clf]['params']['hog_channel'], 
                                           spatial_feat=self.classifiers[clf]['params']['spatial_feat'], 
                                           hist_feat=self.classifiers[clf]['params']['hist_feat'], 
                                           hog_feat=self.classifiers[clf]['params']['hog_feat'])   
                on_windows.extend(curr_on_windows)
        print(on_windows)
        return on_windows
    
    
    def get_on_windows(self, image):
        draw_image = np.copy(image)
        on_windows = []
#         all_windows = []
        window_params = list(zip(self.x_start_stop, 
                                 self.y_start_stop, 
                                 self.xy_window, 
                                 self.xy_overlap))
        for el_x_start_stop, el_y_start_stop, el_xy_window, el_xy_overlap in window_params:
            windows = self.slide_window(image, 
                                   x_start_stop = el_x_start_stop, 
                                   y_start_stop = el_y_start_stop, 
                                   xy_window = el_xy_window, 
                                   xy_overlap = el_xy_overlap)
        
#             all_windows.extend([windows])
        
            curr_on_windows = self.search_windows(image, 
                                       windows=windows, 
                                       clf=self.svc, 
                                       scaler=self.scaler, 
                                       color_space=self.color_space, 
                                       spatial_size=self.spatial_size, 
                                       hist_bins=self.hist_bins, 
                                       orient=self.orient, 
                                       pix_per_cell=self.pix_per_cell, 
                                       cell_per_block=self.cell_per_block, 
                                       hog_channel=self.hog_channel, 
                                       spatial_feat=self.spatial_feat, 
                                       hist_feat=self.hist_feat, 
                                       hog_feat=self.hog_feat)   
            on_windows.extend(curr_on_windows)
        return on_windows
    
    
    
    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
        # Return updated heatmap
        return heatmap# Iterate through list of bboxes
        
    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap
    
    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img
    
    def get_window_coordinates(self):
        w_size = [64, 128, 192, 256]
        w_overlap = 0.8
        self.xy_window = [ (w, w) for w in w_size ]
        self.xy_overlap = [ (w_overlap, w_overlap) for _ in w_size ]
        self.x_start_stop = [ [None, None] for _ in w_size ]
        self.y_top = [ 410, 400, 395, 405 ]
        self.y_start_stop = [[y, int(y+w/2)] for y,w in zip(self.y_top, w_size)]

    
    def detect_vehicles(self, image):
        draw_image = np.copy(image)
        on_windows = []
        all_windows = []
    
        window_params = list(zip(self.x_start_stop, 
                                 self.y_start_stop, 
                                 self.xy_window, 
                                 self.xy_overlap))
        for el_x_start_stop, el_y_start_stop, el_xy_window, el_xy_overlap in window_params:
            windows = self.slide_window(image, 
                                   x_start_stop = el_x_start_stop, 
                                   y_start_stop = el_y_start_stop, 
                                   xy_window = el_xy_window, 
                                   xy_overlap = el_xy_overlap)
    
            all_windows.extend([windows])
    
            curr_on_windows = self.search_windows(image, 
                                       windows=windows, 
                                       clf=self.svc, 
                                       scaler=self.scaler, 
                                       color_space=self.color_space, 
                                       spatial_size=self.spatial_size, 
                                       hist_bins=self.hist_bins, 
                                       orient=self.orient, 
                                       pix_per_cell=self.pix_per_cell, 
                                       cell_per_block=self.cell_per_block, 
                                       hog_channel=self.hog_channel, 
                                       spatial_feat=self.spatial_feat, 
                                       hist_feat=self.hist_feat, 
                                       hog_feat=self.hog_feat)   
            on_windows.extend(curr_on_windows)
    
        window_img = self.draw_boxes(draw_image, on_windows, color=(0, 255, 255), thick=4)                    
    
        colors = [ (255, 0, 0), 
                   (0, 255, 0), 
                   (0, 0, 255), 
                   (0, 255, 255) ]
        for idx, el_windows in enumerate(all_windows):
            draw_image = self.draw_boxes(draw_image, el_windows, color=colors[idx % len(colors)], thick=6)     
        
        return window_img, draw_image

    
    # Define a function to return HOG features and visualization
    def get_hog_features(self,
                         img, 
                         orient, 
                         pix_per_cell, 
                         cell_per_block, 
                         vis=False, 
                         feature_vec=True):
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
    def bin_spatial(self, 
                    img, 
                    size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel() 
        # Return the feature vector
        return features
    
    # Define a function to compute color histogram features 
    # NEED TO CHANGE bins_range if reading .png files with mpimg!
    def color_hist(self,
                   img, 
                   nbins=32, 
                   bins_range=(0, 256)):
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
    def extract_features(self, 
                         imgs, 
                         color_space='RGB', 
                         spatial_size=(32, 32),
                         hist_bins=32, 
                         orient=9, 
                         pix_per_cell=8,
                         cell_per_block=2, 
                         hog_channel=0,
                         spatial_feat=True, 
                         hist_feat=True, 
                         hog_feat=True):
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
    
    
    
    
    
    
        
    def slide_window(self, img, 
                     x_start_stop=[None, None], 
                     y_start_stop=[None, None], 
                     xy_window=(64, 64), 
                     xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        ny_windows = np.int(yspan/ny_pix_per_step) - 1
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list
    
    # Define a function to draw bounding boxes
    def draw_boxes(self, 
                   img, 
                   bboxes, 
                   color=(0, 0, 255), 
                   thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy
    
    # Define a function to extract features from a single image window
    # This function is very similar to extract_features()
    # just for a single image rather than list of images
    def single_img_features(self, 
                            img, 
                            color_space='RGB', 
                            spatial_size=(32, 32),
                            hist_bins=32, 
                            orient=9, 
                            pix_per_cell=8, 
                            cell_per_block=2, 
                            hog_channel=0,
                            spatial_feat=True, 
                            hist_feat=True, 
                            hog_feat=True):    
        #1) Define an empty list to receive features
        img_features = []
        #2) Apply color conversion if other than 'RGB'
#         if color_space != 'RGB':
#             if color_space == 'HSV':
#                 feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#             elif color_space == 'LUV':
#                 feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
#             elif color_space == 'HLS':
#                 feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#             elif color_space == 'YUV':
#                 feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#             elif color_space == 'YCrCb':
#                 feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
#         else: feature_image = np.copy(img)      
        
        if color_space == 'RGB':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        
        #3) Compute spatial features if flag is set
        if spatial_feat == True:
            spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = self.color_hist(feature_image, nbins=hist_bins)
            #6) Append features to list
            img_features.append(hist_features)
        #7) Compute HOG features if flag is set
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(self.get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))      
            else:
                hog_features = self.get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            #8) Append features to list
            img_features.append(hog_features)
    
        #9) Return concatenated array of features
        return np.concatenate(img_features)
    
    # Define a function you will pass an image 
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(self, 
                       img, 
                       windows, 
                       clf, 
                       scaler, 
                       color_space='RGB', 
                       spatial_size=(32, 32), 
                       hist_bins=32, 
                       hist_range=(0, 256), 
                       orient=9, 
                       pix_per_cell=8, 
                       cell_per_block=2, 
                       hog_channel=0, 
                       spatial_feat=True, 
                       hist_feat=True, 
                       hog_feat=True):
    
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
            #4) Extract features for that window using single_img_features()
            features = self.single_img_features(test_img, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
            #5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = clf.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows
        
