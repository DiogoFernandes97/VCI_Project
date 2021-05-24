import cv2 as cv
import numpy as np
import os.path as fl
from matplotlib import pyplot as plt

class Piece:
    def __init__(self, width, height,color_index,center_x,center_y):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.color_index = color_index

    def widthStr(self):
        return str(self.width)
    def heightStr(self):
        return str(self.height)
    def dimsStr(self):
        return str(self.widthStr()+"x"+self.heightStr())


def mmToBlocks(dim_mm):
    dim_blocks = 0
    dim_blocks = round((dim_mm + 0.2)/8)
    if dim_blocks > 3:
        dim_blocks = round(dim_blocks/2)*2
    return dim_blocks