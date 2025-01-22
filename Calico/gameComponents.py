import numpy as np
from gridTypes import HexGrid

class Tile:
    def __init__(self, params):
        self.pattern = params['pattern']
        self.color = params['color']

class Board:
    def __init__(self, gridDetails):
        self.grid = HexGrid((gridDetails['length'],gridDetails['width']),gridDetails['filled'])
        #TODO Finish this