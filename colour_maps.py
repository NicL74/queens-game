# Some predefined colour maps for the game
# These maps can be used to represent different game states or configurations
import numpy as np

# Example board states for an 8x8 chessboard
colour_map_8_1 = np.array([ [0, 0, 0, 1, 1, 1, 1, 1],
                            [0, 2, 2, 2, 2, 1, 1, 1],
                            [0, 3, 3, 3, 2, 1, 1, 1],
                            [0, 3, 0, 2, 2, 2, 1, 1],
                            [0, 0, 0, 0, 2, 4, 4, 4],
                            [0, 5, 0, 0, 2, 6, 4, 6],
                            [5, 5, 0, 0, 2, 6, 6, 6],
                            [5, 5, 5, 5, 5, 6, 7, 6]])

# Test for complete column of single colour
colour_map_8_2 = np.array([ [0, 0, 0, 1, 1, 1, 1, 1],
                            [0, 2, 2, 2, 2, 1, 1, 1],
                            [0, 3, 3, 3, 2, 1, 1, 1],
                            [0, 3, 0, 2, 2, 2, 1, 1],
                            [0, 0, 0, 0, 2, 4, 4, 4],
                            [0, 5, 0, 0, 2, 6, 4, 6],
                            [0, 5, 0, 0, 2, 6, 6, 6],
                            [0, 5, 5, 5, 5, 6, 7, 6]])

# Test for complete row and column of single colour
colour_map_8_3 = np.array([ [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 2, 2, 1, 1, 1],
                            [0, 3, 3, 3, 2, 1, 1, 1],
                            [0, 3, 0, 2, 2, 2, 1, 1],
                            [0, 0, 0, 0, 2, 4, 4, 4],
                            [0, 5, 0, 0, 2, 6, 4, 6],
                            [0, 5, 0, 0, 2, 6, 6, 6],
                            [0, 5, 5, 5, 5, 6, 7, 6]])