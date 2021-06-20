import cv2 as cv
import numpy as np
import os.path as fl
#from matplotlib import pyplot as plt

class Range:
    def __init__(self):
        self.num_range = 0
        self.name = []
    def parseFile(self,path):
        f = open(path, "r")
        nonempty_lines = [line.strip("\n") for line in f if line != "\n"]
        line_count = len(nonempty_lines)
        f.close()
        if line_count % 3 != 0:  # para garantir que temos todos os valores necessarios
            return -1
        else:
            self.num_range = line_count // 3
        
        
        self.lower_value = np.zeros([self.num_range, 3])
        self.upper_value = np.zeros([self.num_range, 3])

        f = open(path, "r")

        for i in range(0, self.num_range):
            self.name.append(f.readline().rstrip(":\n"))

            x = ((f.readline().lstrip("[")).rstrip("]\n")).split()
            self.lower_value[i, 0] = x[0]
            self.lower_value[i, 1] = x[1]
            self.lower_value[i, 2] = x[2]

            x = ((f.readline().lstrip("[")).rstrip("]\n")).split()
            self.upper_value[i, 0] = x[0]
            self.upper_value[i, 1] = x[1]
            self.upper_value[i, 2] = x[2]

    def checkInRange(self,value):
        for k in range(0, self.num_range):
            if np.all(cv.inRange(value, self.lower_value[k,:],self.upper_value[k,:]) == 255 ):
                return k
        
        return 0
    def getName(self, color_index):
        return self.name[color_index]
