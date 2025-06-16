import pygame
import math
from gridTypes import *
from scoring import PointTile
import json
# Constants
HEX_SIZE = 30  # Radius of hexagon
WIDTH = 800
HEIGHT = 600
BACKGROUND_COLOR = (255, 255, 255)

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
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = pygame.mouse.get_pos()
                    hx, hy = pixel_to_hex(mx, my, HEX_SIZE)

                    # Place tile with a sample color pattern
                    hexgrid[(hx+grid_shape[0], hy)] = Tile({'pattern':1, 'color':3}) 
    pygame.quit()

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
display_hexgrid(grid)