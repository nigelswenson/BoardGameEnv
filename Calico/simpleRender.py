import pygame
import math
from gridTypes import *
from scoring import PointTile
import json
from gameComponents import GameManger
from controllers import HumanController
# Constants
HEX_SIZE = 30  # Radius of hexagon
WIDTH = 800
HEIGHT = 600
BACKGROUND_COLOR = (255, 255, 255)
GAME_PHASES = ['store','inventory','placement']
STORE_LOCATIONS = [[1-6,2], [0-6,4], [-1-6,6]]
INVENTORY_SPOTS = [[0-6,9], [2-6,9], [4-6,9]]
pattern_key = {0:(0,0,255),
               1:(255,0,0),
               2:(0,255,0),
               3:(0,255,255),
               4:(255,0,255),
               5:(255,255,0),
               6:(0,0,0)}

def hex_corner(center, size, i):
    angle_deg = 60 * i - 30
    angle_rad = math.radians(angle_deg)
    return (center[0] + size * math.cos(angle_rad),
            center[1] + size * math.sin(angle_rad))

def draw_hexagon(surface, center, size, top_color, bottom_color):
    # Get hexagon corners
    corners = [hex_corner(center, size, i) for i in range(6)]

    # Top half (3 points: top left, top, top right)
    top_half = [corners[0], corners[1], corners[2], center]
    # Bottom half (3 points: bottom right, bottom, bottom left)
    bottom_half = [corners[3], corners[4], corners[5], center]

    pygame.draw.polygon(surface, top_color, top_half)
    pygame.draw.polygon(surface, bottom_color, bottom_half)
    pygame.draw.polygon(surface, (0, 0, 0), corners, 1)  # Outline

def hex_to_pixel(x, y, size):
    """Converts hex coordinates to pixel positions (pointy-topped layout)"""
    px = size * math.sqrt(3) * (x + 0.5 * (y))
    py = size * 3/2 * y+50
    return (int(px), int(py))

def pixel_to_hex(px, py, size):
    """Inverse of your hex_to_pixel (for pointy-topped hexes with offset)"""
    py -= 50  # Undo the offset first
    y = 2/3 * py / size
    x = np.round(px/math.sqrt(3) / size - 0.5*y)
    y = np.round(y)
    return x,y

def display_hexgrid(hexgrid):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Hex Grid Display")
    clock = pygame.time.Clock()
    grid_shape = hexgrid.shape()
    running = True
    while running:
        screen.fill(BACKGROUND_COLOR)
        
        for (x, y), tile in hexgrid:
            center = hex_to_pixel(x-grid_shape[0], y, HEX_SIZE)
            if type(tile) is Tile:
                
                draw_hexagon(screen, center, HEX_SIZE, pattern_key[tile.pattern], pattern_key[tile.color])
            elif tile =='Points':
                draw_hexagon(screen, center, HEX_SIZE, pattern_key[6], pattern_key[6])
        pygame.display.flip()
        clock.tick(30)

        # Handle quit
        for event in pygame.event.get():
            print(event)
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = pygame.mouse.get_pos()
                    hx, hy = pixel_to_hex(mx, my, HEX_SIZE)
                    # Place tile with a sample color pattern
                    print(grid_shape[0])
                    hexgrid[(hx+grid_shape[0], hy)] = Tile({'pattern':1, 'color':3}) 
    pygame.quit()

def play_game(hexgrid):
    pygame.init()
    pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.Font('freesansbold.ttf', 32)

    # create a text surface object,
    # on which text is drawn on it.
    white = (255, 255, 255)
    black = (0, 0, 0)
    text1 = font.render('Inventory', True, black, white)
    textRect1 = text1.get_rect()
    text2 = font.render('Store', True, black, white)
    textRect2 = text2.get_rect()

    # create a rectangular object for the
    # text surface object
    

    # set the center of the rectangular object.

    pygame.display.set_caption("Calico Game")
    clock = pygame.time.Clock()
    grid_shape = hexgrid.shape()
    running = True
    game_phase = 'store'
    selected_tile = None
    free_location = 2
    controller = HumanController(
                                zone1_coords=STORE_LOCATIONS,
                                zone2_coords=INVENTORY_SPOTS,
                                hex_size=HEX_SIZE,
                                grid_shape=hexgrid.shape()
                                )
    game_board = GameManger(1,controllers=[controller])
    hexgrid = game_board.players[0].board.grid
    text3 = font.render(f'Cat 1 L {game_board.players[0].board.cat_functions[0].cluster_point_threshold}', True, black, white)
    textRect3 = text3.get_rect()
    text4 = font.render(f'Cat 2 L {game_board.players[0].board.cat_functions[1].cluster_point_threshold}', True, black, white)
    textRect4 = text3.get_rect()
    text5 = font.render(f'Cat 3 L {game_board.players[0].board.cat_functions[2].cluster_point_threshold}', True, black, white)
    textRect5 = text3.get_rect()
    textRect1.center = (300, 400)
    textRect2.center = (100, 100)
    textRect3.center = (170, 500)
    textRect4.center = (380, 500)
    textRect5.center = (600, 500)
    c1_ids = game_board.players[0].board.cat_functions[0].ids
    c2_ids = game_board.players[0].board.cat_functions[1].ids
    c3_ids = game_board.players[0].board.cat_functions[2].ids
    hexgrid[[-8,11]] = Tile({'pattern':c1_ids[0], 'color':c1_ids[1]})
    hexgrid[[-4,11]] = Tile({'pattern':c2_ids[0], 'color':c2_ids[1]})
    hexgrid[[0,11]] = Tile({'pattern':c3_ids[0], 'color':c3_ids[1]})
    for i in range(3):
        hexgrid[STORE_LOCATIONS[i]] = game_board.store[i]
    hexgrid[INVENTORY_SPOTS[0]] = game_board.players[0].tiles[0]
    hexgrid[INVENTORY_SPOTS[1]] = game_board.players[0].tiles[1]
    for _ in range(22):
        screen.fill(BACKGROUND_COLOR)
        screen.blit(text1, textRect1)
        screen.blit(text2, textRect2)
        screen.blit(text3, textRect3)
        screen.blit(text4, textRect4)
        screen.blit(text5, textRect5)
        for (x, y), tile in hexgrid:
            center = hex_to_pixel(x-grid_shape[0], y, HEX_SIZE)
            if type(tile) is Tile:
                
                draw_hexagon(screen, center, HEX_SIZE, pattern_key[tile.pattern], pattern_key[tile.color])
            elif isinstance(tile,PointTile):
                draw_hexagon(screen, center, HEX_SIZE, pattern_key[6], pattern_key[6])
        pygame.display.flip()
        clock.tick(30)
        game_board.takeTurn()
        print('done with turn',game_board.players[0].points)
        for i in range(3):
            hexgrid[STORE_LOCATIONS[i]] = game_board.store[i]
        hexgrid[INVENTORY_SPOTS[0]] = game_board.players[0].tiles[0]
        hexgrid[INVENTORY_SPOTS[1]] = game_board.players[0].tiles[1]
        # Handle quit
    print('game finished')

with open('CalicoBoardset1.json') as jsonFile:
    boardInfo = json.load(jsonFile)
new_details = []
for row in boardInfo:
    temp = []
    for item in row:
        if type(item) == dict:
            temp.append(Tile(item))
        else:
            temp.append(item)
    new_details.append(temp)
grid = HexGrid(new_details)
# display_hexgrid(grid)
play_game(grid)