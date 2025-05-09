import pygame

# Constantes dynamiques
WIDTH           = 1200
HEIGHT          =  800
CARD_WIDTH      = WIDTH * 0.04
CARD_HEIGHT     = HEIGHT * 0.10
MARGIN          = WIDTH * 0.01
GRID_ROWS       = 3
GRID_COLS       = 4
FPS             = 60
AUTO_PLAY_DELAY = 0.05
MAX_POINTS      = 100


# Couleurs
WHITE        = (255, 255, 255)
GRAY         = (180, 180, 180)
RED          = (255, 0, 0)
DARK_GRAY    = (80, 80, 80)
BLACK        = (0, 0, 0)
GREEN_TABLE  = (100, 100, 100)
YELLOW       = (255, 255, 0)
VALUE_COLORS = {
    'neg' : (0, 0, 139),
    'zero': (135, 206, 250),
    'low' : (0, 200, 0),
    'mid' : (255, 255, 0),
    'high': (200, 0, 0)
}
