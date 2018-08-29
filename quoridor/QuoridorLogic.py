from collections import deque
import numpy as np
import time
import pathFinder

def is_wall_legal(moves):
    if moves == None: return 0
    if moves == [0,1]: return 1
    return moves[0].is_wall_legal(moves[1:])

class Board():

    def __init__(self, n):
        "Set up initial board configuration."
        self.n = n * 2 - 1
        self.n_= n - 1
        # Create the empty board array.
        self.pieces = np.zeros((4, self.n, self.n), dtype='uint8')

        self.pieces[0][self.n-1][self.n/2] = 1
        self.pieces[1][0][self.n/2] = 1
        pathFinder.setup(n)

    def get_legal_moves(self, color, pool=None):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        """
        moves = [0]*8
        (x,y) = self.player_position(color)
        (x_,y_) = self.player_position(-color)
        block = np.add(self.pieces[2], self.pieces[3])

        # up
        if (x>0 and not block[x-1][y]):
            if (x-2==x_ and y==y_):                                                 # other player is blocking
                if (x-4>=0 and not block[x-3][y]):
                    moves[0] = 1                                                    # jump over
                else:
                    if (y+2<=self.n-1 and not block[x_][y_+1]):
                        moves[7] = 1                                                # diagonal
                    if (y-2>=0 and not block[x_][y_-1]):
                        moves[4] = 1                                                # diagonal
            else:
                moves[0] = 1                                                        # normal 1 step

        # down
        if (x<self.n-1 and not block[x+1][y]):
            if (x+2==x_ and y==y_):
                if (x+4<=self.n-1 and not block[x+3][y]):
                    moves[1] = 1
                else:
                    if (y+2<=self.n-1 and not block[x_][y_+1]):
                        moves[5] = 1
                    if (y-2>=0 and not block[x_][y_-1]):
                        moves[6] = 1
            else:
                moves[1] = 1

        # left
        if (y>0 and not block[x][y-1]):
            if (x==x_ and y-2==y_):
                if (y-4>=0 and not block[x][y-3]):
                    moves[2] = 1
                else:
                    if (x-2>=0 and not block[x_-1][y_]):
                        moves[4] = 1
                    if (x+2<=self.n-1 and not block[x_+1][y_]):
                        moves[6] = 1
            else:
                moves[2] = 1

        # right
        if (y<self.n-1 and not block[x][y+1]):
            if (x==x_ and y+2==y_):
                if (y+4<=self.n-1 and not block[x][y+3]):
                    moves[3] = 1
                else:
                    if (x-2>=0 and not block[x_-1][y_]):
                        moves[7] = 1
                    if (x+2<=self.n-1 and not block[x_+1][y_]):
                        moves[5] = 1
            else:
                moves[3] = 1
        #t0 = time.time()
        #wall_moves_1 = self.wall_moves(color, pool)
        #t1 = time.time()
        wall_moves_2 = self.wall_moves_(color)
        #t2 = time.time()

        #print (t2-t1)*1000, " vs. ", (t1-t0)*1000
        #if wall_moves_1 != wall_moves_2:
            #print "False"
            #print self.pieces
            #print wall_moves_1
            #print wall_moves_2
            #print repr(self.pieces.tobytes())

        moves += wall_moves_2
        #moves += self.wall_moves(color, pool)
        return moves

    def player_position(self, color):
        idx = 0 if color==1 else 1
        pos = np.argmax(self.pieces[idx])
        return (pos/self.n, pos%self.n)

    def wall_moves_(self, color):
        idx = 2 if color==1 else 3
        if (np.sum(self.pieces[idx])==20):
            return [0]*(self.n_**2)*2
        return pathFinder.legalWalls(self.pieces.tobytes())

    def wall_moves(self, color, pool):
        idx = 2 if color==1 else 3
        if (np.sum(self.pieces[idx])==20):
            return [0]*(self.n_**2)*2

        moves = [None]*(self.n_**2)*2
        moves_boolean = np.zeros((self.n_**2)*2, dtype='uint8')
        counter = -1
        block = np.add(self.pieces[2], self.pieces[3])
        block_ = list(block.ravel())

        # horizontal walls
        for x in range(1,self.n,2):
            for y in range(0,self.n-2,2):
                counter += 1
                if (not block[x][y] and not block[x][y+2] and ((not block[x-1][y+1] or not block[x+1][y+1]) or (x-3>=0 and x+3<self.n and block[x-1][y+1] and block[x-3][y+1] and block[x+1][y+1] and block[x+3][y+1]))):
                    moves[counter] = [self, 8, [x,y], block_]
                    moves_boolean[counter] = 1
        # vertical walls
        for x in range(2,self.n,2):
            for y in range(1,self.n,2):
                counter += 1
                if (not block[x][y] and not block[x-2][y] and ((not block[x-1][y-1] or not block[x-1][y+1]) or (y-3>=3 and y+3<self.n and block[x-1][y-1] and block[x-1][y-3] and block[x-1][y+1] and block[x-1][y+3]))):
                    moves[counter] = [self, 9, [x,y], block_]
                    moves_boolean[counter] = 1
        # clean illegal walls
        # pre-pass: find a path and mark all non-blocking walls as valid
        if np.sum(moves_boolean) > len(moves_boolean)/3:
            parent, tail = self.has_a_path(color, block, True)
            parent_,tail_ = self.has_a_path(-color, block, True)
            if parent != 0 and parent_ !=0:
                valid_walls = self.walls_not_in_path(parent,tail) & self.walls_not_in_path(parent_,tail_) & moves_boolean
                for i in range(len(valid_walls)):
                    if valid_walls[i]:
                        moves[i] = [0,1]

        if pool is None:
            return [self.is_wall_legal(move[1:]) if move != None else 0 for move in moves]
        return pool.map(is_wall_legal, moves)

    def is_wall_legal(self, move):
        if move == None or move == [0]: return 0
        if move == [1]: return 1
        w,(x,y),block_ = move
        block = np.array(block_).reshape((self.n,self.n))

        # place the wall
        block[x][y]=1
        if (w==8 and y+2<self.n):
            block[x][y+2]=1
        elif (w==9 and x-2>=0):
            block[x-2][y]=1

        # look for a path
        legality = self.has_a_path(1, block) and self.has_a_path(-1, block)
        return legality

    def has_a_path(self, color, block, return_path = False):
        idx = 0 if color==1 else 1
        goal_x = 0 if color==1 else self.n-1
        (x,y) = self.player_position(color)
        parent = None
        if return_path:
            parent = {str(x)+','+str(y): None}
        visited = np.zeros((self.n,self.n), dtype='uint8')
        visited[x][y] = 1
        next_cells = deque()
        #next_cells = self.destination_cells_from(x, y, block, visited, parent)
        self.destination_cells_from(x, y, block, visited, parent, next_cells)

        while len(next_cells)>0:
            tmp = next_cells.popleft()
            if tmp[0]==goal_x:
                if return_path:
                    return parent, str(tmp[0])+','+str(tmp[1])
                return 1
            visited[tmp[0]][tmp[1]] = 1
            #next_cells_from_here = self.destination_cells_from(tmp[0], tmp[1], block, visited, parent)
            #next_cells += next_cells_from_here
            self.destination_cells_from(tmp[0], tmp[1], block, visited, parent, next_cells)
        if return_path:
            return 0,0
        return 0

    def destination_cells_from(self, x, y, block, visited, parent, cells):
        #cells = []
        if parent == None:  # minor optimization
            if (x>0 and not block[x-1][y] and not visited[x-2][y]):
                cells.append([x-2, y])  # up
            if (x<self.n-1 and not block[x+1][y] and not visited[x+2][y]):
                cells.append([x+2, y])  # down
            if (y>0 and not block[x][y-1] and not visited[x][y-2]):
                cells.append([x, y-2])  # left
            if (y<self.n-1 and not block[x][y+1] and not visited[x][y+2]):
                cells.append([x, y+2])  # right
            #return cells
            return

        parent_xy = str(x) + ',' + str(y)
        if (x>0 and not block[x-1][y] and not visited[x-2][y]):
            cells.append([x-2, y])  # up
            parent[str(x-2)+','+str(y)] = parent_xy
        if (x<self.n-1 and not block[x+1][y] and not visited[x+2][y]):
            cells.append([x+2, y])  # down
            parent[str(x+2)+','+str(y)] = parent_xy
        if (y>0 and not block[x][y-1] and not visited[x][y-2]):
            cells.append([x, y-2])  # left
            parent[str(x)+','+str(y-2)] = parent_xy
        if (y<self.n-1 and not block[x][y+1] and not visited[x][y+2]):
            cells.append([x, y+2])  # right
            parent[str(x)+','+str(y+2)] = parent_xy
        #return cells

    def walls_not_in_path(self, parent, tail):
        valids = np.ones((self.n_**2)*2, dtype='uint8')

        while parent[tail] != None:
            head = parent[tail]
            x, y = tail.split(',')
            x, y = int(x), int(y)
            x_, y_ = head.split(',')
            x_, y_ = int(x_), int(y_)
            tail = head
            # vertical move only blocked by horizontal walls
            if y == y_:
                ax, ay = (x+x_)/2, y
                if ay<self.n -1:
                    valids[self.index_of_action(8, ax, ay)-8] = 0
                if ay-2>=0:
                    valids[self.index_of_action(8, ax, ay-2)-8] = 0
            # horizontal move only blocked by vertical walls
            if x == x_:
                ay, ax = (y+y_)/2, x
                if ax>=2:
                    valids[self.index_of_action(9, ax, ay)-8] = 0
                if ax+2<=self.n-1:
                    valids[self.index_of_action(9, ax+2, ay)-8] = 0

        return valids

    def execute_move(self, move, color):
        a, (x,y) = self.action_from_index(move, color)
        # moving
        idx = 0 if color==1 else 1
        if (a<8):
            self.pieces[idx].fill(0)
            self.pieces[idx][x][y]=1
            return
        # placing a wall
        idx = 2 if color==1 else 3
        if (a==8):
            self.pieces[idx][x][y]=1
            self.pieces[idx][x][y+2]=1
        if (a==9):
            self.pieces[idx][x][y]=1
            self.pieces[idx][x-2][y]=1

    def index_of_action(self, a, x, y):
        if (a<8): return a                                          # 0 - 7
        if (a==8):
            return 7 + (x/2)*self.n_ + (y/2)+1                      # 8 - 71
        if (a==9):
            return 7 + (self.n_**2) + (y+1)/2 + ((x/2)-1)*self.n_   # 72 - 135

    def action_from_index(self, index, color):
        if (index<8):
            return self.move_action_destination(index, color)
        if (7 < index < 8+(self.n_**2)):
            c = index - 8
            return [8, [(c / self.n_)*2 + 1, (c % self.n_)*2]]
        if (7+(self.n_**2) < index < 8+2*(self.n_**2)):
            c = index - 8 - (self.n_**2)
            return [9, [(c / self.n_)*2 + 2, (c % self.n_)*2 + 1]]

    def move_action_destination(self, action, color):

        (x,y) = self.player_position(color)
        (x_,y_) = self.player_position(-color)
        block = np.add(self.pieces[2], self.pieces[3])

        # up
        if (action==0):
            if (x>0 and not block[x-1][y]):
                if (x-2==x_ and y==y_):                                                                   # other player is blocking
                    if (x-4>=0 and not block[x-3][y]):
                        return([0, [x-4,y]])                                                              # jump over
                else:
                    return([0, [x-2,y]])                                                                  # normal 1 step
        # down
        if (action==1):
            if (x<self.n-1 and not block[x+1][y]):
                if (x+2==x_ and y==y_):
                    if (x+4<=self.n-1 and not block[x+3][y]):
                        return([1, [x+4,y]])
                else:
                    return([1, [x+2,y]])
        # left
        if (action==2):
            if (y>0 and not block[x][y-1]):
                if (x==x_ and y-2==y_):
                    if (y-4>=0 and not block[x][y-3]):
                        return([2, [x, y-4]])
                else:
                    return([2, [x,y-2]])
        # right
        if (action==3):
            if (y<self.n-1 and not block[x][y+1]):
                if (x==x_ and y+2==y_):
                    if (y+4<=self.n-1 and not block[x][y+3]):
                        return([3, [x, y+4]])
                else:
                    return([3, [x,y+2]])
        # up-left
        if (action==4):
            return ([4, [x-2,y-2]])
        # down-right
        if (action==5):
            return ([5, [x+2,y+2]])
        # down-left
        if (action==6):
            return ([6, [x+2,y-2]])
        # up-right
        if (action==7):
            return ([7, [x-2,y+2]])

        print ('\nnothing found')
        print (action,color)
        print self.pieces[0]
        print self.pieces[1]
        print self.pieces[2]
        print self.pieces[3]
