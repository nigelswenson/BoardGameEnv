import numpy as np
from gridTypes import HexGrid, Tile
import json
import copy 
from collections import defaultdict
from scoring import *

class Board:
    def __init__(self, gridDetails, pointFunctions):
        new_details = []
        self.hexes = []
        for row in gridDetails:
            temp = []
            for item in row:
                if type(item) == dict:
                    temp.append(Tile(item))
                else:
                    temp.append(item)
            new_details.append(temp)
        self.grid = HexGrid(new_details)
        i=0
        for k,v in self.grid:
            if v == 'Points':
                self.grid[k] = pointFunctions['hexes'][i]
                i+=1
                self.hexes.append(k)
        self.openSpaces = []
        self.buildSpaces()
        self.pins = [] # list of (tile group, color, points)
        self.cats = [] # list of (tile group, cat, points)
        self.cat_functions = pointFunctions['cats']
        
        self.hex_functions = pointFunctions['hexes']
        self.hex_score = 0
        self.pin_score = 0
        self.cat_score = 0
        self.rainbow_pin=False
        self.color_groups = {0:set(),1:set(),2:set(),3:set(),4:set(),5:set()}
        self.pin_scoring = BasicClusterManager()
        for location, value in self.grid:
            if type(value) is Tile:
                self.pin_scoring.addTile(value.color,location,self.grid.getAdjacent(location))
                for cat in self.cat_functions:
                    cat.addTile(value.pattern,location,self.grid.getAdjacent(location))
        self.starting_board = copy.deepcopy(self.grid)
        self.starting_cats = copy.deepcopy(self.cat_functions)
        self.starting_hex = copy.deepcopy(self.hex_functions)
        self.starting_pins = copy.deepcopy(self.pin_scoring)

    def buildSpaces(self):
        # iterates through the hex grid and finds the spaces without tiles
        self.openSpaces=[]
        for key, value in self.grid:
            if value=="None":
                self.openSpaces.append(key)
        # if len(self.openSpaces) ==0:
        #     print('All Spaces filled, game should be over')

    def placeTile(self,tile,location):
        if location not in self.openSpaces:
            print(self.openSpaces)
            raise IndexError('Desired Location is already filled')
        else:
            self.grid[location] = tile
            self.pin_scoring.addTile(tile.color,location,self.grid.getAdjacent(location))
            for cat in self.cat_functions:
                cat.addTile(tile.pattern,location,self.grid.getAdjacent(location))
        self.buildSpaces()
        
    def checkCats(self):
        # check over the board for new cats to be added
        self.cat_score = 0
        for i in range(3):
            cat_clusters = self.cat_functions[i].getPointClusters()
            for _, _, cluster_size in cat_clusters:
                if cluster_size>= self.cat_functions[i].cluster_point_threshold:
                    self.cat_score += self.cat_functions[i].catValue

    def checkPins(self):
        point_clusters = self.pin_scoring.getPointClusters()
        for location, value, cluster_size in point_clusters:
            if cluster_size>= 3:
                self.color_groups[value].add(location)
        lens = np.array([len(grouping) for grouping in self.color_groups.values()])
        if all(lens>=1):
            self.rainbow_pin=True
        self.pin_score = self.rainbow_pin*3 + sum(lens)*3

    def checkHexes(self):
        # check over the board for hex points
        self.hex_score = 0
        for hex_key in self.hexes:
            self.grid[hex_key].calculate_score(self.grid.getAdjacent(hex_key))
            self.hex_score += self.grid[hex_key].get_score()
    
    def getScore(self):
        # print(f"hex score: {self.hex_score}, pin score: {self.pin_score}, cat score: {self.cat_score}")
        return self.cat_score+self.pin_score+self.hex_score

    def reset(self):
        self.grid = copy.deepcopy(self.starting_board)
        self.cat_functions = copy.deepcopy(self.starting_cats)
        self.hex_functions = copy.deepcopy(self.starting_hex)
        self.pin_scoring = copy.deepcopy(self.starting_pins)
        # BasicClusterManager()
        self.color_groups = {0:set(),1:set(),2:set(),3:set(),4:set(),5:set()}
        self.rainbow_pin = False
        # for location, value in self.grid:
        #     if type(value) is Tile:
        #         self.pin_scoring.addTile(value.color,location,self.grid.getAdjacent(location))
        #         for cat in self.cat_functions:
        #             cat.addTile(value.pattern,location,self.grid.getAdjacent(location))
        self.buildSpaces()

class Store:
    def __init__(self):
        with open('CalicoTileset.json') as jsonFile:
            temp = json.load(jsonFile)
        self.tileCollection = []
        for t1 in temp:
            self.tileCollection.append(Tile(t1))
            self.tileCollection.append(Tile(t1))
            self.tileCollection.append(Tile(t1))
        self.backupTiles = copy.deepcopy(self.tileCollection)
        np.random.shuffle(self.tileCollection)
        self.openTiles = [self.tileCollection.pop() for _ in range(3)]
        

    def selectTile(self, ind):
        try:
            new_tile = self.tileCollection.pop()
        except IndexError:
            new_tile = None
        temp = self.openTiles[ind]
        self.openTiles[ind] = new_tile
        return temp

    def reset(self):
        self.tileCollection = copy.deepcopy(self.backupTiles)
        np.random.shuffle(self.tileCollection)
        self.openTiles = [self.tileCollection.pop() for _ in range(3)]
        
class Player:
    def __init__(self, startingTiles, gridNum, hexes, controller, cats):
        self.tiles = startingTiles
        with open(f'CalicoBoardset{gridNum+1}.json') as jsonFile:
            boardInfo = json.load(jsonFile)
        i = 0

        self.board = Board(boardInfo, {'cats':cats,'hexes':hexes})
        self.hexFunctions = hexes
        self.controller = controller
        self.points = 0


    def takeTurn(self, store):
        # needs to grab a tile. Logic for this can be handled by 
        selected = self.controller.selectTile(store,self.board)
        self.tiles.append(selected)
        tile, placement = self.controller.placeTile(self.tiles, self.board)
        temp = self.tiles.pop(tile)
        self.board.placeTile(temp,placement)
        self.board.checkHexes()
        self.board.checkPins()
        self.board.checkCats()
        self.points = self.board.getScore()
        # self.checkHexPoints()

hexset_key={'AAABBB':AAABBBTile,
 'AAAABB':AAAABBTile,
 'AABBCC':AABBCCTile,
 'AAABBC':AAABBCTile,
 'AABBCD':AABBCDTile,
 'NotEqual':NotEqualTile}

class GameManger:
    def __init__(self, numPlayers, controllers, catsUsed=None, hexesUsed=None):
        self.store = Store()

        if hexesUsed is None:
            with open('CalicoHexset.json') as jsonFile:
                temp = json.load(jsonFile)
            selections = np.random.choice(list(temp.keys()),3,replace=False)
            self.hexes = []
            for key in selections:
                self.hexes.append(hexset_key[key](temp[key]))
        else:
            self.hexes=hexesUsed
        if catsUsed is None:
            with open('CalicoReducedCatset.json') as jsonFile:
                temp = json.load(jsonFile)
            selections = np.random.choice(list(temp.values()),3,replace=False)
            orders = list(range(6))
            np.random.shuffle(orders)
            self.catsUsed = [CatClusterManager(sel['size'],orders[i*2:i*2+2],sel['points']) for i,sel in enumerate(selections)]
        else:
            self.catsUsed=catsUsed
        self.players = [Player([self.store.tileCollection.pop(),self.store.tileCollection.pop()],
                                i, self.hexes, controllers[i],self.catsUsed) for i in range(numPlayers)]


        self.turnsLeft=22
    
    def takeTurn(self):
        for player in self.players:
            player.takeTurn(self.store, self.catsUsed)
        self.turnsLeft -= 1

    def runGame(self):
        while self.turnsLeft>0:
            self.takeTurn()
        scores = [p.points for p in self.players]
        winner = np.argsort(scores)
        print(f'Player Number {winner[0]+1} is the winner with a total score of {scores[winner[0]]}')
        print(f'All Player scores')
        for i,p in enumerate(scores):
            print(f'Player {i+1}: {p} points')