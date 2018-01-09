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

from video_handler import VideoHandler

def process_video(args):
    print("Starting video processing")
    for key, value in (vars(args)).items():
        print("{:15s} -> {}".format(key, value))
    
    video_handler = VideoHandler(args)
    print("video_handler: {}".format(video_handler.clip_name))
    
    video_handler.process_video()

def main():
    description = "Vehicle Detection Python project" 
    time_now = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    parser = argparse.ArgumentParser(description=description)
#     parser.add_argument("-i", "--input_video", type=str, default='project_video.mp4')
    parser.add_argument("-i", "--input_video", type=str, default='test_video.mp4')
#     parser.add_argument("-o", "--output_video", type=str, default='project_video_out_{}.mp4'.format(time_now))
    parser.add_argument("-o", "--output_video", type=str, default='test_video_out.mp4')
#     parser.add_argument("-s", "--subclip_length", type=int, default=7)
    args = parser.parse_args(sys.argv[1:])
    
    print("args == ")
    print(args)
    process_video(args)
    
if __name__ == '__main__':
    main()