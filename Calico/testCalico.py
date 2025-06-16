from gameComponents import *
from gridTypes import HexGrid
from controllers import randomSelection
# store = Store()

# p1 = Player([store.tileCollection.pop(),store.tileCollection.pop()]
#             , 1, None, None)
controller= randomSelection()
gm = GameManger(1,[controller])
gm.runGame()
# print(p1.tiles)
# print(len(p1.board.openSpaces))
# print(p1.board.grid.data.keys())
# p1.board.checkPins()
# p1.board.placeTile(Tile({'pattern':0,'color':0}),(-1,5))
# p1.board.checkPins()
# p1.board.placeTile(Tile({'pattern':0,'color':0}),(-2,5))
# p1.board.checkPins()


