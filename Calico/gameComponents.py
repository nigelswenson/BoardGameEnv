import numpy as np
from gridTypes import HexGrid
import json

class Tile:
    def __init__(self, params):
        self.pattern = params['pattern']
        self.color = params['color']

class Board:
    def __init__(self, gridDetails):
        self.grid = HexGrid(gridDetails)
        #TODO Finish this
        self.openSpaces = []
        self.buildSpaces()

    def buildSpaces(self):
        # iterates through the hex grid and finds the spaces without tiles
        self.openSpaces=[]
        for key, value in self.grid:
            if value is None:
                self.openSpaces.append(key)
        if len(self.openSpaces) ==0:
            print('All Spaces filled, game should be over')

    def placeTile(self,tile,location):
        if location in self.openSpaces:
            raise IndexError('Desired Location is already filled')
        else:
            self.grid[location] = tile
        self.buildSpaces()

class Store:
    def __init__(self):
        with open('CalicoTileset.json') as jsonFile:
            temp = json.loads(jsonFile)
        self.tileCollection = [Tile(t1) for t1 in temp]
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


class Player:
    def __init__(self, startingTiles, gridNum, hexes, controller):
        self.tiles = startingTiles
        with open(f'CalicoBoardset{gridNum}.json') as jsonFile:
            boardInfo = json.loads(jsonFile)
        self.board = Board(boardInfo)
        self.cats = []
        self.buttons = []
        self.hexPoints = []
        self.hexFunctions = hexes
        self.controller = controller
        self.points = 0

    def takeTurn(self, store, cats):
        # needs to grab a tile. Logic for this can be handled by 
        selected = self.controller.selectTile(store,self.board)
        self.tiles.append(selected)
        tile, placement = self.controller.placeTile(self.tiles, self.board)
        temp = self.tiles.pop(tile)
        self.board[placement] = temp
        self.checkCats(catFunctions=cats)
        self.checkButtons()
        self.checkHexPoints()

    def checkCats(self, catFunctions):
        # iterate over the board to check if any cat has been attracted. needs cat function
        pass
    def checkButtons(self):
        # iterate over the board to check if any button should be added or removed
        pass
    def checkHexPoints(self):
        # iterate over the board to check if any of the three hexes get points
        pass
class pointForPattern:
    def __init__(self, pattern, score):
        self.pattern = pattern
        self.score = score
    

class GameManger:
    def __init__(self, numPlayers, controllers, catsUsed=None, hexesUsed=None):
        self.store = Store()
        if hexesUsed is None:
            with open('CalicoHexset.json') as jsonFile:
                temp = json.loads(jsonFile)
            selections = np.random.choice(range(6),3,replace=False)
            self.hexes = pointForPattern([temp[i] for i in selections])
        else:
            self.hexes=hexesUsed
        self.players = [Player([self.store.tileCollection.pop(),self.store.tileCollection.pop()],
                                i, self.hexes, controllers[i]) for i in range(numPlayers)]
        if catsUsed is None:
            with open('CalicoCatset.json') as jsonFile:
                temp = json.loads(jsonFile)
            selections = np.random.choice(range(9),3,replace=False)
            self.catsUsed = pointForPattern([temp[i] for i in selections])
        else:
            self.catsUsed=catsUsed

        self.turnsLeft=20
    
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