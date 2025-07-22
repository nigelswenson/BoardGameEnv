import numpy as np
import pygame

def pixel_to_hex(px, py, size):
    """Inverse of your hex_to_pixel (for pointy-topped hexes with offset)"""
    py -= 50  # Undo the offset first
    y = 2/3 * py / size
    x = np.round(px/np.sqrt(3) / size - 0.5*y)
    y = np.round(y)
    return x,y

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

class HumanController:
    def __init__(self, zone1_coords, zone2_coords, hex_size, grid_shape):
        self.zone1 = zone1_coords
        self.zone2 = zone2_coords
        self.zone2_free = 2
        self.hex_size = hex_size
        self.grid_shape = grid_shape  # needed for coordinate translation
        self.selected_tile_key = None  # keep track of selected tile in placeTile
    
    def wait_for_click_on_zone(self, zone):
        # print('waiting for click in zone')
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    hx, hy = pixel_to_hex(mx, my, self.hex_size)
                    hx += self.grid_shape[0]  # normalize + offset if needed

                    if [int(hx), int(hy)] in zone:
                        for i,value in enumerate(zone):
                            if value == [int(hx), int(hy)]:
                                return i
                    else:
                        print('invalid selection', zone,[int(hx), int(hy)])
                        print((hx, hy) in zone)
            pygame.time.wait(10)  # short delay to avoid CPU hogging
    
    def wait_for_click_on_board(self, valid_spaces):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    hx, hy = pixel_to_hex(mx, my, self.hex_size)
                    hx += self.grid_shape[0]
                    if (int(hx),int(hy)) not in valid_spaces:
                        print('invalid selection')
                        print(f'you selected {hx, hy}, valid are {valid_spaces}')
                    else:
                        # Accept any hex coordinate not necessarily in zones
                        return (int(hx), int(hy))
            pygame.time.wait(10)

    def selectTile(self, store, board):
        # store is the hexgrid dict or same structure with tiles in Zone 1
        # Let user click on one tile in Zone 1
        tile_ind = self.wait_for_click_on_zone(self.zone1)
        selected = store.selectTile(tile_ind)
        # board.grid[self.zone2[self.zone2_free]]= selected
        return selected
        # self.selected_tile_key = pos  # Save the key for possible removal or identification
        # return tile

    def placeTile(self, tiles, board):
        # User first clicks a hex in zone 2 (to "capture" color/pattern)
        zone2_pos = self.wait_for_click_on_zone(self.zone2)
        self.zone2_free = zone2_pos
        # print(self.zone2[2])
        # print(board.grid[self.zone2[2]])
        # board.grid[self.zone2[zone2_pos]] = None
        
        # Then user clicks anywhere on board to place using that color/pattern
        board_pos = self.wait_for_click_on_board(board.openSpaces)

        # Find tile index to place from tiles list (assuming tiles have pattern/color)
        # This part assumes the player wants to place from the tiles list, match by pattern/color or just pick first
        # tile_index = None
        # for i, t in enumerate(tiles):
        #     if t.pattern == pattern and t.color == color:
        #         tile_index = i
        #         break
        # else:
        #     # fallback: just put index 0 (or handle error)
        #     tile_index = 0

        return zone2_pos, board_pos
    


class randomSelection:
    def selectTile(self, store, boardstate):
        return store.selectTile(np.random.randint(3))
    
    def placeTile(self, tiles, board):
        board_ind = np.random.choice(range(len(board.openSpaces)))
        return np.random.randint(3), board.openSpaces[board_ind]