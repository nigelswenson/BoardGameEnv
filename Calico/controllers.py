import numpy as np

class humanInput:
    def selectTile(self, store, boardstate):
        valid = False
        while not valid:
            a = input('select a tile (1-4)')
            try:
                playerInput = int(a)
                if (playerInput >-1) and (playerInput) < 5:
                    picked = store.selectTile(playerInput-1)
                    valid = True
            except:
                print('please input a valid tile')
        return picked
    
    def placeTile(self, tiles, board):
        valid = False
        while not valid:
            a = input('select a tile (1-3)')
            try:
                playerInput = int(a)
                if (playerInput >-1) and (playerInput) < 4:
                    picked = playerInput-1
                    valid = True
            except:
                print('please input a valid tile')
        valid=False
        while not valid:
            a = input(f'select a board location (1-{len(board.openSpaces)}')
            try:
                playerInput = int(a)
                placed = playerInput -1
            except:
                print('please input a valid tile')
        return picked, board.openSpaces[placed]
    
class randomSelection:
    def selectTile(self, store, boardstate):
        return store.selectTile(np.random.randint(4))
    
    def placeTile(self, tiles, board):
        return np.random.randint(3), np.random.choice(board.openSpaces)