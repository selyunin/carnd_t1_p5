## Vehicle Detection Project

### The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction
  on a labeled training set of images and train a classifier Linear
  SVM classifier
* Optionally, you can also apply a color transform and append binned
  color features, as well as histograms of color, to your HOG feature
  vector. 
* Note: for those first two steps don't forget to normalize your
  features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier
  to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4
  and later implement on full project_video.mp4) and create a heat
  map of recurring detections frame by frame to reject outliers and
  follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/image_1.png
[image2]: ./output_images/image_2.png
[image3]: ./output_images/image_3.png
[image4]: ./output_images/image_4.png
[image5]: ./output_images/image_5.png
[image6]: ./output_images/image_6.png
[image7]: ./output_images/image_7.png
[image8]: ./output_images/image_8.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. The Writeup file includes all the rubric points and how each one has been addressed.

Note that the `jyputer-notebook` P5.ipynb also provides a thorough
description of the project.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

First I did a short dataset exploration: downloaded the data, 
and extracted the archives in the corresponding directories.
The datasets are formed as a combination of data from several sources
(i.e. from KITTI and GTI datasets). I then read all image names from
all the sub-directories for vehicles and non-vehicles and print the
summary (see the `P5.ipynb` for more detailed output).
Each image in the dataset is a 64x64 color image:

I then print random images of vehicles as well as non-vehicles:

![alt text][image1]

![alt text][image2]

I then split the data into training, validation and test sets,  using
`train_test_split` function from `sklearn.cross_validation` to first split
the data into training and (validation and test) sets. We then split
validation and test sets. Note, then in order to detect what images
are misclassified later, we both divide and shuffle image names.
Before, I read images using cv2.imread, but in this way after
shuffling it is not possible to identify what images were
misclassified.

In this project I use histogram of oriented gradients, the `hog` function from
`skimage.feature` to find orientations of gradients in each cell of a
specified size of image. Function `get_hog_features` takes image, size
of a cell, cells per block and returns hog features of the image.
Function `bin_spatial` takes image and returns spatial features (i.e.
contiguous flattened array as image). Function `color_hist` creates
color features (i.e. concatenates histograms of all color channels). 

The function `extract_features` combines different feature extraction.
Given an image name, we first read the file (we use `cv2.imread` and
obtain *BGR* image, but in range 0 to 255. The problem with
`matplotlib.image` is that it reads `*.png` files to the domain
`[0,1]` and not to `[0, 255]`). We then convert the color space to the
specified one, extract spatial and color features, if specified, and
return all concatenated features.


#### 2. Explain how you settled on your final choice of HOG parameters.

Below are the example parameters we use to train the classifier. As we see
in the implementation for video stream, we use two classifiers and
then combine their predictions. 
The one mentioned here performs pretty decent.
The parameters has been found empirically, by choosing the parameters,
and then evaluating the classifier.


| Parameter        | Value         |
|:----------------:|:-------------:| 
| color_space      | 'YUV'         | 
| orient           | 11            | 
| pix_per_cell     | 8             | 
| cell_per_block   | 2             | 
| hog_channel      | 'ALL'         | 
| spatial_size     | (16, 16)      | 
| hist_bins        | 32            | 
| spatial_feat     | True          | 
| hist_feat        | True          | 
| hog_feat         | True          | 


Below I show hog features for  vehicles and non-vehicles from the dataset:


![alt text][image3]


![alt text][image4]


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

For training, validation and test data I then extracted features with
the specified parameters (see the table above). I then stack all the features 
into one vector and scale the features, in order to equalize impact of
different features.

As I used all available data for scaling, I then retrieved back pieces for
training, validation and test.

The classification problem in the project is binary -- either vehicle
or non-vehicle. For vehicles our desired output is `1` and for
non-vehicles is `0`. I prepared the corresponding y-values. 
Additionally, I also created a random state and shuffle both features 
and the corresponding image names, 
to be able to backtrack what images are classified incorrectly.

I then used LinearSVC end trained the classifier using prepared training
data,  then check  the results on validation and test sets.

In the Jyputer Notebook I also explored what the classifier has predicted wrong in the
validation set: identified false positives (images that depict
non-vehicles, but classifier marked them as vehicles) and false
negatives (images of vehicles that were marked as non-vehicles by the
classifier) and visualized the results.

False positives and false negatives look as follows:

![alt text][image5]


![alt text][image6]


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?


When camera is mounted on a vehicle and records an image of a road,
vehicles appear in different locations of the frame and in a different
scale (depending on a distance). To account for this, we use sliding
window search, where image is split in cells of different size and
location (alternatively, several windows of different size are slided
on an image and vehicle detection is performed in each image.

The function `slide_window` returns a list of windows that are within
the pre-specified start and stop coordinates in `x` and `y` and
overlap by a given ratio. The function `draw_boxes` gets an image and
a list of boxes coordinates and draws rectangles with the specified
coordinates. `single_img_features` extracts features from a patch of
an image. `search_windows` returns a list of windows, for which the
classifier predicted presense of *vehicles*.

Since it only make sense to search for vehicles in the region of a
road, we define a method `get_window_coordinates` that identifies
coordinates to restricts the search to the specified region of an
image:

`x_start_stop : [[None, None], [None, None], [None, None], [None, None]]`

`y_start_stop : [[410, 442], [400, 464], [395, 491], [405, 533]]`

`xy_window    : [(64, 64), (128, 128), (192, 192), (256, 256)]`

`xy_overlap   : [(0.8, 0.8), (0.8, 0.8), (0.8, 0.8), (0.8, 0.8)]`

The method `detect_vehicles` does the following: for each window size,
start and stop coordinates, overlap ratio search for vehicles and
obtain *on_windows* (where the classifier predicted presence of
vehicle). We also draw *all_windows*, in which we searched for
vehicles. We return two images, the one on which *on_windows* are
shown, and the other one on which *all_windows* are shown.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

