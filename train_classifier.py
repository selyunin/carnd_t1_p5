#!/usr/bin/env python

'''
Created on Jan 6, 2018
@author: selyunin
'''

import os
import sys
import cv2
import glob
import numpy as np
import datetime
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from collections import OrderedDict

from classifier_fit import ClassifierFit

def train_classifier(args):
    print("Starting LinearSVM fit")
    for key, value in (vars(args)).items():
        print("{:15s} -> {}".format(key, value))
    
    classifier = ClassifierFit(args)
    print("ClassifierFit: {}".format(classifier))
    
def main():
    description = "Vehicle Detection Python project" 
    time_now = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input_video", type=str, default='test_video.mp4')
    parser.add_argument("-o", "--output_video", type=str, default='test_video_out.mp4')
    parser.add_argument("-s", "--subclip_length", type=int, default=7)
    args = parser.parse_args(sys.argv[1:])
    
    print("args == ")
    print(args)
    train_classifier(args)
    
if __name__ == '__main__':
    main()