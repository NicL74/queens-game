from time import time
import numpy as np
import matplotlib.pyplot as plt
import colour_maps  # Assuming colour_maps.py is in the same directory

# Define some variables
board_size = 8
test = 1  # This variable is not used in the code but can be used for testing purposes

# Class to represent a game, comprising a colour map and a queen map and take board_size as an argument
# The colour map represents the colours of the squares on the board
# The queen map represents the presence of queens on the board, with different values for different states
# 0 = empty, 1 = queen, 2 = cross, 3 = invalid queen
# The class has methods to load a predefined board state, set the state of a square, toggle the state of a square, visualize the board, count colours, and count entries            

class Game(object):
    def __init__(self, board_size=8):
        self.colour_map = np.zeros((board_size, board_size))
        self.queen_map = np.zeros((board_size, board_size))    
        self.colours_count = np.zeros(board_size, dtype=int)
        self.entries_count = np.zeros(3, dtype=int)  # Assuming 3 types of entry: queens, crosses, errors
        self.entries_per_colour = np.zeros((board_size, 3), dtype=int)  # 3 columns for queens, crosses, and errors
        
    def load_board(self):
        print("Loading predefined board state...")
        # Example board state for an 8x8 chessboard
        self.colour_map = colour_maps.colour_map_8_1  # Load a predefined colour map
        # If you want to use a different colour map, you can change this line
        # For example, to use colour_map_8_2 or colour_map_8_3, you can uncomment the following lines:
        #self.colour_map = colour_maps.colour_map_8_2
        #self.colour_map = colour_maps.colour_map_8_3
        # Convert the colour map to integers for consistency
        self.colour_map = self.colour_map.astype(int)
        # Initialize the queen map to zeros
        # This will represent an empty board with no queens placed
        # The queen map will be used to track the state of each square
        # 0 = empty, 1 = queen, 2 = cross, 3 = invalid queen
        # Here we initialize the queen_map to zeros, indicating no queens are placed
        # This will be updated as the game progresses
        self.queen_map = np.zeros((board_size, board_size))
        return self.colour_map.astype(int), self.queen_map.astype(int)  
        
    def set_square(self, row, col, queen_value):
        if 0 <= row < board_size and 0 <= col < board_size:
            self.queen_map[row, col] = queen_value
        else:
            raise ValueError("Row and column must be within the bounds of the board.")
        
    def toggle_square(self, row, col):
        if 0 <= row < board_size and 0 <= col < board_size:
            if self.queen_map[row, col] == 0:
                self.queen_map[row, col] = 2  # Set to cross state
            elif self.queen_map[row, col] == 2:
                self.queen_map[row, col] = 1  # Set to queen state
            elif self.queen_map[row, col] == 1:
                self.queen_map[row, col] = 0  # Reset to empty state
        else:
            raise ValueError("Row and column must be within the bounds of the board.")
    
    def visualize_board(self):
        plt.figure(figsize=(board_size, board_size))
        plt.imshow(self.colour_map, cmap='gist_rainbow', vmin=0, vmax=board_size)
        #plt.colorbar()
    
        # Add black grid lines between squares but no ticks
        plt.gca().set_xticks(np.arange(-0.5, board_size, 1), minor=True)        
        plt.gca().set_yticks(np.arange(-0.5, board_size, 1), minor=True)
        plt.gca().grid(which='minor', color='black', linestyle='-', linewidth=2)
 
    
        # Overlay queens on the board
        for i in range(board_size):
            for j in range(board_size):
                if self.queen_map[i, j] == 1:
                    plt.text(j, i, '♛', fontsize=32, ha='center', va='center', color='black')
                if self.queen_map[i, j] == 2:
                    plt.text(j, i, 'x', fontsize=32, ha='center', va='center', color='black')
                if self.queen_map[i, j] == 3:
                    plt.text(j, i, '♛', fontsize=40, ha='center', va='center', color='black')
                    plt.text(j, i, '♛', fontsize=32, ha='center', va='center', color='red')
        
        plt.xticks(np.arange(board_size), np.arange(1, board_size + 1))
        plt.yticks(np.arange(board_size), np.arange(1, board_size + 1))
        plt.grid(False)
        plt.ion()
        plt.show()  
        plt.pause(1)
        
    # Function to count haw many entries in the colour_map are equal to each value 0 to board_size-1
    def count_colours(self,print_counts=False):
        for i in range(board_size):
            for j in range(board_size):
                if self.colour_map[i, j] < board_size:
                    self.colours_count[self.colour_map[i, j]] += 1
                    
        if print_counts:
            print("Colour counts: ", self.colours_count)
        return

    # Function to count the number of queens on the board
    def count_entries(self,print_counts=False):
        self.entries_count = [0, 0, 0] # Reset counts
        # For each square in the queen_map, count the number of queens, crosses and errors
        for i in range(board_size):
            for j in range(board_size):
                if self.queen_map[i, j] == 1:
                    self.entries_count[0] += 1  # Queens that have not been invalidated
                elif self.queen_map[i, j] == 2:
                    self.entries_count[1] += 1  # Crosses
                elif self.queen_map[i, j] == 3:
                    self.entries_count[2] += 1  # Invalid queens
        if print_counts:
            print("Entries counts (queens, crosses, errors): ", self.entries_count)
        return  

    # Function to count the number of entries per colour
    def count_entries_per_colour(self,print_counts=False):   
        # Reset the entries_per_colour array
        self.entries_per_colour.fill(0)

        # For each colour, count the number of queens, crosses and errors
        for i in range(board_size):
            for j in range(board_size):
                if self.queen_map[i, j] == 1:
                    self.entries_per_colour[self.colour_map[i, j], 0] += 1    # Count queens      
                elif self.queen_map[i, j] == 2:
                    self.entries_per_colour[self.colour_map[i, j], 1] += 1  # Count crosses
                elif self.queen_map[i, j] == 3:
                    self.entries_per_colour[self.colour_map[i, j], 2] += 1  # Count errors
        if print_counts:
            print("Entries counts per colour (queens, crosses, errors): ", self.entries_per_colour)
        return  

    # Function to check which colours have only one queen and then add to every other square of that colour
    def check_single_queen_per_colour(self,print_values=False): 
        # Create array of row, col values for any queens added
        queen_positions = []
        # Check if the queen_map is empty
        #if np.all(self.queen_map == 0):
        #    raise ValueError("Queen map is empty. Please load a board state first.")
        
        # Call count_entries_per_colour to get the counts
        self.count_entries_per_colour(print_counts=print_values)        
        # For each colour, check if it has only one queen
        for colour_index in range(self.entries_per_colour.shape[0]):
            queen_count = self.entries_per_colour[colour_index, 0]
            if print_values:
                print("Colour index: ", colour_index, " Count of queens: ", queen_count)
            # If a colour has only one queen, set all other squares of that colour to cross state
            if queen_count == 1:
                # Find the squares of this colour
                squares = np.argwhere(self.colour_map == colour_index)
                for square in squares:
                    row, col = square
                    if self.queen_map[row, col] == 0:  # Only set if the square is empty
                        self.queen_map[row, col] = 2  # Set to cross state
                        queen_positions.append((row, col))
        if print_values:
            print("Final queen map after checking single queens per colour:\n", self.queen_map) 
        return queen_positions     
        
    # Function to fill in all crosses for a single queen just been added
    def fill_in_crosses_for_single_queen(self, queen_location, print_values=False):
        # Check if the queen_location is valid
        if not isinstance(queen_location, tuple) or len(queen_location) != 2:
            raise ValueError("Queen location must be a tuple of (row, col).")
        row, col = queen_location
        if not (0 <= row < board_size and 0 <= col < board_size):
            raise ValueError("Row and column must be within the bounds of the board.")
        # Call count_entries_per_colour to get the counts
        counts = self.entries_per_colour
        
        # For each value in counts, print the list number and the count of queens   
        if counts.shape[1] < 3:
            raise ValueError("Counts array must have at least 3 columns for queens, crosses, and errors.")
        # Enumerate through the counts array
        for colour_index in range(counts.shape[0]):
            queen_count = counts[colour_index, 0]
            if print_values:
                print("Colour index: ", colour_index, " Count of queens: ", queen_count)
            # If a colour has only one queen, set all other squares of that colour to cross state
            if queen_count == 1:
                # Find the squares of this colour
                squares = np.argwhere(self.colour_map == colour_index)
                for square in squares:
                    row, col = square
                    if self.queen_map[row, col] == 0:  # Only set if the square is empty
                        self.queen_map[row, col] = 2  # Set to cross state
        if print_values:
            print("Final queen map after checking single queens per colour:\n", self.queen_map)

        return 
    
    def check_for_full_row_or_column(self, print_values=False):
        # Check for full rows or columns of colours_map to see if any row or column is completely filled with a single colour
        result = np.array(['Type', 'Index', 'Colour'])
        for i in range(board_size):
            if np.all(self.colour_map[i, :] == self.colour_map[i, 0]):
                if print_values:
                    print(f"Row {i} is completely filled with colour {self.colour_map[i, 0]}")
                result = np.vstack((result, ['row',i,self.colour_map[i, 0]]))
            if np.all(self.colour_map[:, i] == self.colour_map[0, i]):
                if print_values:
                    print(f"Column {i} is completely filled with colour {self.colour_map[0, i]}")
                result = np.vstack((result, ['col',i,self.colour_map[0, i]]))
        if print_values:
            print("result.shape: ",result.shape)
            print("result: ", result)
        if result.shape != (3,): # i.e. if it has more than just the header row
            return result[1:]  # Return the result excluding the header row
        else:
            return None
    
    def fill_in_single_colour_squares(self):
        # Create array of row, col values for any queens added
        queen_positions = []
        # Find all squares that are empty and have a single colour in the colour_map
        for i in range(board_size):
            for j in range(board_size):
                if self.queen_map[i, j] == 0:
                    # Check if the square has a single colour in the colour_map
                    if np.sum(self.colour_map[i, j] == self.colour_map) == 1:
                        self.queen_map[i, j] = 1  # Set to queen state
                        success = True
                        queen_positions.append((i, j))
        return queen_positions
    
        
    def add_crosses_after_full_row_or_column(self, print_values=False):
        # Set success flag
        success = False
        # Check for full rows or columns of colours_map to see if any row or column is completely filled with a single colour
        result = self.check_for_full_row_or_column()
        if result is not None:
            # Result can include at most one row and one column
            row_present = False
            col_present = False
            row = None
            col = None
            colour = None
            for entry in result:
                entry_type, index, colour = entry
                #print("colour",colour)
                #print(type(colour))
                if entry_type == 'row':
                    row = int(index)
                    row_present = True
                    if print_values:
                        print("row", row)
                if entry_type == 'col':
                    col = int(index)
                    col_present = True
                    if print_values:
                        print("col",    col)

            # If result row and column are both present, we can fill in a queen at the intersection
            if row_present and col_present:
                if self.queen_map[row, col] == 0:  # Only set if the square is empty
                    self.queen_map[row, col] = 1  # Set to queen state
                    success = True
                    # Call the check_single_queen_per_colour method to ensure only one queen per colour
                    self.check_single_queen_per_colour(print_values=print_values)
                    # Call the fill_in_crosses_for_single_queen method to fill in crosses for the single queen just been added
                    self.fill_in_crosses_for_single_queen((row, col), print_values=print_values)
                    if print_values:
                        print(f"Added queen at ({row}, {col}) for colour {colour}")
                    return success

            # Find all squares of the specified colour except for the row or column that is full
            squares = np.argwhere(self.colour_map == int(colour))
            #print("squares",squares)
            for square in squares:
                r, c = square
                if entry_type == 'row' and r != row:
                    if self.queen_map[r, c] == 0:  # Only set if the square is empty
                        self.queen_map[r, c] = 2  # Set to cross state
                        success = True
                if entry_type == 'col' and c != col:
                    if self.queen_map[r, c] == 0:  # Only set if the square is empty
                        self.queen_map[r, c] = 2  # Set to cross state
                        success = True
        else:
            pass
        return success

    def check_for_colours_limited_to_single_row_or_column(self, print_values=False):
        # Step through all the colours  
        for colour in range(board_size):
            # Create an array containing 1 where the colour is present and queen_map is 0 and zero otherwise    
            pass
    # Function to create a 'queen kernal function' shadow mask
    # i.e. binary matrix where 1s represent squares that are threatened by a queen at (row, col)
    def create_qkf_shadow_mask(self, row, col):
        mask = np.zeros_like(self.queen_map)
        # Threaten all squares in the same row and column
        mask[row, :] = 1
        mask[:, col] = 1
        # Threaten all squares within 1 square
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= row + i < board_size and 0 <= col + j < board_size:
                    mask[row + i, col + j] = 1
        return mask.astype(int)

    def get_all_empty_squares_of_colour(self, colour):
        empty_squares = np.zeros_like(self.queen_map)
        for r in range(board_size):
            for c in range(board_size):
                if self.colour_map[r, c] == colour and self.queen_map[r, c] == 0:
                    empty_squares[r, c] = 1
        return empty_squares.astype(int)

    def check_if_square_should_get_crossed(self, row, col, print_values=False):
        # Get colour of current square
        my_colour = self.colour_map[row, col]
        # Create qkf_mask
        qkf_mask = self.create_qkf_shadow_mask(row, col)
        if print_values:
            print(f"Queen Kernel Function Shadow Mask for ({row}, {col}):")
            print(qkf_mask)
        # Run through all the colours except for my_colour
        for colour in range(board_size):
            if colour == my_colour:
                continue
            # Get all empty squares of the current colour
            empty_squares = self.get_all_empty_squares_of_colour(colour)
            if print_values:
                print(f"Empty squares for colour {colour}:")
                print(empty_squares)
            # Check if zero empty squares and return false if so
            if np.sum(empty_squares) == 0:
                return False
            # Perform logical AND of the empty squares with the qkf_mask
            affected_squares = empty_squares * qkf_mask
            if print_values:
                print(f"Affected squares for colour {colour}:")
                print(affected_squares)
            # Subtract affected squares from empty squares and sum the result               
            diff = np.sum(affected_squares - empty_squares)
            if print_values:
                print(f"Difference for colour {colour}: {diff}")
            if diff == 0:
                print(f"Square ({row}, {col}) should get crossed for colour {colour}")
                return True
        return False

    def check_if_all_squares_should_get_crossed(self):
        crossed = False
        # Check all squares on the board
        for row in range(board_size):
            for col in range(board_size):
                # Check if the square is empty
                if self.queen_map[row, col] == 0:
                    if self.check_if_square_should_get_crossed(row, col, print_values=False):
                        # Put a cross in that square
                        self.queen_map[row, col] = 2  # Set to cross state
                        crossed = True
        return crossed

    def print_queen_map(self):
        print("Queen Map:")
        for row in self.queen_map:
            print(' '.join(str(int(x)) for x in row))
                
    def print_colour_map(self):
        print("Colour Map:")
        for row in self.colour_map:
            print(' '.join(str(int(x)) for x in row))

    # Main function to run the game logic
def main():
    game1 = Game(8)  # Create a game instance with a board size of 8
    
    # Call entry counting function
    game1.count_entries(print_counts=True)  # Count entries and print the counts
    
    # Load a predefined board
    game1.load_board()  # Load the predefined board state
    
    # Call colour counting method
    game1.count_colours(print_counts=True)  # Count colours and print the counts
    

    
    if(test == 1):
        # Set some example squares
        #game1.set_square(0, 0, 1)  # Set square (0,0) to queen
        #game1.set_square(1, 1, 2)  # Set square (1,1) to cross
        #game1.set_square(2, 2, 3)  # Set square (2,2) to invalid queen   
        #game1.set_square(1, 5, 1)  # Set square (0,0) to queen
        #game1.set_square(0, 0, 2)
        #game1.set_square(0, 1, 2)
        #game1.set_square(0, 2, 2)   
        
        # Call the function to create a queen kernel function shadow mask
        #qkf_mask = game1.create_qkf_shadow_mask(0, 0)
        #print("Queen Kernel Function Shadow Mask:")
        #print(qkf_mask)
        
        #qkf_mask = game1.create_qkf_shadow_mask(3, 1)
        #print("Queen Kernel Function Shadow Mask:")
        #print(qkf_mask)
        
        #qkf_mask = game1.create_qkf_shadow_mask(4, 4)
        #print("Queen Kernel Function Shadow Mask:")
        #print(qkf_mask)
        
        #qkf_mask = game1.create_qkf_shadow_mask(7, 4)
        #print("Queen Kernel Function Shadow Mask:")
        #print(qkf_mask)
        pass
        
    # Call entry counting method
    game1.count_entries(print_counts=True)  # Count entries and print the counts
    
    # Call count_entries_per_colour method    
    game1.count_entries_per_colour(print_counts=True)  # Count entries per colour and print the counts
    
    # Call check for full row or column method
    result = game1.check_for_full_row_or_column(print_values=True)  # Check for full rows or columns
    print("Result of full row or column check:\n", result)  # Print the result of the check
    
    # Call check for single queens per colour method
    added_queens = game1.check_single_queen_per_colour(print_values=True)  # Check for single queens per colour and update the queen map
    print("Added queens for single colours: ", added_queens)  # Print the positions of added queens

    # Call fill in crosses for single queen
    for row,col in added_queens:
        game1.fill_in_crosses_for_single_queen(row, col)

    # Call count_entries_per_colour method
    game1.count_entries_per_colour(print_counts=True)  # Count entries per colour and print the counts
    
    # Call add_crosses_after_full_row_or_column method
    success = game1.add_crosses_after_full_row_or_column(print_values=True)  # Add crosses after checking for full rows or columns
    print ("Success of adding crosses after full row or column check: ", success)  # Print success status
 
    # Call fill_in_single_colour_squares method
    added_queens = game1.fill_in_single_colour_squares()  # Fill in single colour squares
    if(added_queens != []):
        print("Added queens for single colours: ", added_queens)  # Print success status

    # Call check for colours limited to single row or column method
    game1.check_for_colours_limited_to_single_row_or_column(print_values=True)

    # Call check_if_all_squares_should_get_crossed method
    all_crossed = game1.check_if_all_squares_should_get_crossed()
    print("All squares that were crossed: ", all_crossed)  # Print if all squares should get crossed
    
    # Visualize the board state
    game1.visualize_board()  # Visualize the board with queens and crosses

    # Pause 2 seconds
    import time
    time.sleep(2)

    # Call check_if_all_squares_should_get_crossed method
    all_crossed = game1.check_if_all_squares_should_get_crossed()
    print("All squares that were crossed: ", all_crossed)  # Print if all squares should get crossed
    
    # Visualize the board state
    game1.visualize_board()  # Visualize the board with queens and crosses
    
    # Print the queen and colour maps
    game1.print_queen_map()  # Print the queen map
    game1.print_colour_map()  # Print the colour map



if __name__ == "__main__":
    main()  
    
# This code initializes a chessboard, loads a predefined state, and visualizes it using matplotlib.
# The queens are represented by '♛' and the squares they threaten by 'x'.