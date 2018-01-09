
'''
Created on Jan 6, 2018
@author: selyunin
'''
# Define a class to receive the characteristics of each line detection
from collections import deque
import numpy as np

class Vehicle():
    def __init__(self):
        self.N_WINDOW = 20
        self.heat  = deque([], self.N_WINDOW)
        self.labels  = deque([], self.N_WINDOW)
        self.current_heat = None
    
    def add_heat(self, heat):
        self.heat.append(heat)
        
    def get_heat(self):
        self.current_heat = sum(self.heat)
        return self.current_heat
    
    def add_labels(self, label):
        self.labels.append(label)
#         for el in self.labels:
#             print(el)
        
    def get_labels(self):
        x = np.zeros_like(self.labels[-1][0])
        l = 0
        for first, second in self.labels:
            x += first
            l += second
        return  x
#         return (self.labels[-1])
#         self.poly_y    = deque([], self.N_WINDOW)
#         self.poly_x    = deque([], self.N_WINDOW)