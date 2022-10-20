import random
import numpy as np
import pandas as pd

class CityGenerator:
    def __init__(self, width, height, nodesNumber):
        self.width = width
        self.height = height
        self.nodesNumber = nodesNumber

    def generate(self):
        cities = pd.read_csv("Data/cities.csv")
        xs = cities['x']
        ys = cities['y']
        gc = cities['GC']
        gm = cities['GM']
        return np.column_stack((xs, ys, gc, gm))