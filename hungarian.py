# This file implements the hungarian maximization algorithm as described here: 
# http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html
import copy

# Zeros in the algorithm can be starred or primed, or neither
NEITHER = 0
STARRED = 1
PRIMED = 2

# Step 1
# For each row, find the min in that row, and subtract it from every cell in
# that row 
def subtractRowMinFromEachRow(grid):
    for r in xrange(len(grid)):
        min_ = min(grid[r])
        for c in xrange(len(grid[r])):
            grid[r][c] -= min_

# Step 2
# For each uncovered zero in the grid, add a star to it and then cover that row and column.
# But then uncover all the rows and cols
def starZeros(grid, grid_info, row_cover, col_cover):
    for r in xrange(len(grid)):
        for c in xrange(len(grid[r])):
            if grid[r][c] == 0 and row_cover[r] == False and col_cover[c] == False:
                grid_info[r][c] = STARRED
                row_cover[r] = col_cover[c] = True

    uncoverAll(row_cover, col_cover)

# Step 3
# Cover all the columns with starred zeros, and return the number of covered columns
def coverColsWithStarredZeros(grid, grid_info, col_cover):
    for r in xrange(len(grid)):
        for c in xrange(len(grid[r])):
            if grid_info[r][c] == STARRED:
                # cover the column
                col_cover[c] = True

    covered_cnt = 0
    for c in xrange(len(col_cover)):
        if col_cover[c]:
            covered_cnt += 1

    return covered_cnt

# Return the column of a starred zero in the specified row, or -1
def isStarInRow(row, grid, grid_info):
    for c in xrange(len(grid[row])):
        if grid_info[row][c] == STARRED:
            return c

    return -1

# Return the row of a starred zero in the specified column, or -1
def isStarInCol(col, grid, grid_info):
    for r in xrange(len(grid)):
        if grid_info[r][col] == STARRED:
            return r

    return -1

# Find an uncovered zero, and return it's row and column (-1 returned if one is not present)
def findZero(grid, grid_info, row_cover, col_cover):
    for r in xrange(len(grid)):
        for c in xrange(len(grid[r])):
            if grid[r][c] == 0 and row_cover[r] == False and col_cover[c] == False:
                return r, c

    return -1, -1

# Step 4
# Keep finding non covered zeros, prime them, and see if there is a starred zero in that row (that you found the non covered zero in).
# If there is no starred zero in that row, return the row and column you just visited, otherwise mark the row as covered, and the column
# you saw the starred zero in as covered
def primeZeros(grid, grid_info, row_cover, col_cover):
    while True:
        r, c = findZero(grid, grid_info, row_cover, col_cover)
        if r == -1:
            return False

        grid_info[r][c] = PRIMED
        col_of_star = isStarInRow(r, grid, grid_info)
        if col_of_star == -1:
            return {'row' : r, 'col' : c}

        row_cover[r] = True
        col_cover[col_of_star] = False

# Find the smallest number in the grid that is not covered
def findSmallestUncoveredValue(grid, grid_info, row_cover, col_cover):
    min_ = None
    for r in xrange(len(grid)):
        for c in xrange(len(grid[r])):
            if row_cover[r] == 0 and col_cover[c] == 0 and (min_ == None or grid[r][c] < min_):
                min_ = grid[r][c]

    return min_

# step 6
# For each number in the grid, if it's row is covered, add the min to its value. 
# If the column is not covered, then subtract the min for it
def applySmallestValue(grid, grid_info, row_cover, col_cover, min_):
    for r in xrange(len(grid)):
        for c in xrange(len(grid[0])):
            if row_cover[r]:
                grid[r][c] += min_
            if not col_cover[c]:
                grid[r][c] -= min_

# Toggle the star in every point in the path
def augmentPath(grid, grid_info, path):
    for pair in path:
        r = pair['row']
        c = pair['col']
        if grid_info[r][c] == STARRED:
            grid_info[r][c] = NEITHER
        else:
            grid_info[r][c] = STARRED

# Remove the prime from every primed item in the path
def unprimeAll(grid_info):
    for r in xrange(len(grid_info)):
        for c in xrange(len(grid_info[r])):
            if grid_info[r][c] == PRIMED:
                grid_info[r][c] = NEITHER

# Uncover all the rows and cols
def uncoverAll(row_cover, col_cover):
    for i in xrange(len(row_cover)):
        row_cover[i] = False
    for i in xrange(len(col_cover)):
        col_cover[i] = False

# Return the column of a primed zero in the specified row, or -1
def isPrimeInRow(row, grid, grid_info):
    for c in xrange(len(grid[row])):
        if grid_info[row][c] == PRIMED:
            return c

    return -1

# Step 5
# I really don't want to explain this. I have no idea what its purpose is. Here is the description:
# Construct a series of alternating primed and starred zeros as follows.  Let Z0 represent the uncovered primed zero found in Step 4.  
# Let Z1 denote the starred zero in the column of Z0 (if any). Let Z2 denote the primed zero in the row of Z1 (there will always be one).  
# Continue until the series terminates at a primed zero that has no starred zero in its column.  
# Unstar each starred zero of the series, star each primed zero of the series, erase all primes and uncover every line in the matrix.  
# Return to Step 3. 
def pathStuff(grid, grid_info, row_cover, col_cover, pair):
    path = []
    path.append(pair)
    done = False
    while not done:
        row = isStarInCol(path[-1]['col'], grid, grid_info)
        if row > -1:
            path.append({"row" : row, "col" : path[-1]['col']})
        else:
            done = True
        if not done:
            col = isPrimeInRow(path[-1]['row'], grid, grid_info)
            path.append({"row" : path[-1]['row'], "col" : col})

    augmentPath(grid, grid_info, path)
    unprimeAll(grid_info)
    uncoverAll(row_cover, col_cover)

# For every element in the grid, set grid[r][c] = max_value_of_grid - grid[r][c]
# This is the trick to turn hungarian minimization into maximization
def normalize(grid):
    max_ = grid[0][0]
    for row in grid:
        max_of_row = max(row)
        if max_of_row > max_:
            max_ = max_of_row

    for r in xrange(len(grid)):
        for c in xrange(len(grid[r])):
            grid[r][c] = max_ - grid[r][c]
    return max_

# Transpose a grid
def transpose(grid):
    new_grid = []
    for c in xrange(len(grid[0])):
        new_grid.append([])
        for r in xrange(len(grid)):
            new_grid[c].append(grid[r][c])

    return new_grid

# The actual hungarian-munkres algorthim
# Returns a matrix where an item is set to 1 if that row, col is part of the extrema
def assign(grid):
    # all rows and cols start off uncovered
    row_cover = [False]*len(grid) 
    col_cover = [False]*len(grid[0])
    # all numbers are non starred, and non primed
    # 1 means it's starred, 2 means it's primed
    grid_info = [[NEITHER]*len(grid[0]) for i in grid]
    # exit when the number of covered zeros equals this
    exit_condition = min(len(grid), len(grid[0])) 
            
    # Step 1
    subtractRowMinFromEachRow(grid)
    # Step 2
    starZeros(grid, grid_info, row_cover, col_cover)
    cover = True
    while True:
        if cover:
            # step 3
            covered_cnt = coverColsWithStarredZeros(grid, grid_info, col_cover)
            if covered_cnt >= exit_condition:
                break
    
        # step 4
        pair = primeZeros(grid, grid_info, row_cover, col_cover)
        if pair:
            # step 5
            pathStuff(grid, grid_info, row_cover, col_cover, pair)
            # go to step 3
            cover = True
        else:
            # step 6
            n = findSmallestUncoveredValue(grid, grid_info, row_cover, col_cover)
            applySmallestValue(grid, grid_info, row_cover, col_cover, n)
            # goto step 4
            cover = False

    return grid_info

# Transpose the matrix (if it has more rows than cols)
# And, if the method is maximization, perform the subtraction rule (the normalization function)
def prepare(grid, method="max"):
    rows = len(grid)
    cols = len(grid[0])
    # The algo requires that the number of cols >= the number of rows
    if rows > cols:
        grid = transpose(grid)
    working_copy = copy.deepcopy(grid)

    # Make the adjustments for maximization 
    if method == "max":
        normalize(working_copy)

    return grid, working_copy

# sum up the starred items in the grid
def sum(grid, grid_info):
    sum = 0
    for r in xrange(len(grid_info)):
        for c in xrange(len(grid_info[r])):
            if grid_info[r][c] == STARRED:
                sum += grid[r][c]

    return sum

# Return a list of the tuples (row, column,) that correspond to the extrema points
def listPoints(grid_info):
    rows = len(grid_info)
    cols = len(grid_info[0])
    # If we transposed the grid earlier, transpose it again to get the proper coordinates
    if rows != cols:
        grid_info = transpose(grid_info)

    points = []
    for r in xrange(len(grid_info)):
        for c in xrange(len(grid_info[r])):
            if grid_info[r][c] == STARRED:
                points.append((r, c))

    return points

# Prepares the matrix for assignment, performs the assignment, and returns the extrema and list of points
# where the extrema occur
def solve(grid, method="min"):
    grid, working_copy = prepare(grid, method)
    grid_info = assign(working_copy)
    return sum(grid, grid_info), listPoints(grid_info)

# convience methods
def minimize(grid):
    return solve(grid, "min") 
def maximize(grid):
    return solve(grid, "max") 
    
