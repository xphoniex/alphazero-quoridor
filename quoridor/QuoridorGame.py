from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .QuoridorLogic import Board
import numpy as np

from multiprocessing import Pool

class QuoridorGame(Game):
    def __init__(self, n):
        self.n = n
        b = Board(self.n)
        self.board = b
        #self.pool = Pool()
        self.pool = None

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return b.pieces

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n*2-1, self.n*2-1)

    def getActionSize(self):
        # return number of actions
        return 8+(2*(self.n-1)**2)

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.n)
        b.pieces = np.copy(board)
        b.execute_move(self.normalizeAction(action, player), player)
        return (b.pieces, -player)

    def print_board(self, board):
        print (board[0])
        print (board[1])
        print (board[2])
        print (board[3])

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        b = Board(self.n)
        b.pieces = np.copy(board)
        valids = b.get_legal_moves(player, self.pool)
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        if (board[0][0].sum()==1):
            return 1
        if (board[1][self.n*2-2].sum()==1):
            return -1
        return 0

    def getCanonicalForm(self, board, player):
        # board from pov of player
        if (player==1):
            return board
        return self.getSymmetries(board,0)

    def normalizeAction(self, action, player):
        # takes index and
        # returns index of mirrored action if player==-1
        if (player==1): return action
        #############################
        if (action<8):
            if (action%2==0):
                return action+1
            else:
                return action-1

        b = Board(self.n)
        a,(x,y) = b.action_from_index(action, -1)
        if (a==8):
            (x,y) = self.normalizePoint(x,y+2)
            return b.index_of_action(8, x, y)
        if (a==9):
            (x,y) = self.normalizePoint(x-2,y)
            return b.index_of_action(9, x, y)

    def normalizePoint(self, x, y):
        size = 2*self.n-1
        board = np.zeros((size, size), dtype='uint8')
        board[x][y] = 1
        board = np.fliplr(np.flipud(board))
        pos = np.argmax(board)
        return (pos/size, pos%size)


    def getSymmetries(self, board, pi):
        pieces = [None]*4
        pieces[0],pieces[1] = np.copy(board[1]),np.copy(board[0])
        pieces[2],pieces[3] = np.copy(board[3]),np.copy(board[2])
        for i in range(4):
            pieces[i] = np.fliplr(np.flipud(pieces[i]))
        return np.array(pieces)

    def stringRepresentation(self, board):
        return board.tostring()

# =======
# display

def placePiece(str, y, color):
    str = list(str)
    str[y] = color
    return "".join(str)

def placeHorizontalWall(str, y, color):
    str = list(str)
    for i in range(y*2, y*2+5):
        str[i] = color
    return "".join(str)

def display(board, swap=1):
    swap = swap == -1
    pieces = ['1','2']
    if swap:
        pieces = ['2','1']
    n = (board[0][0].shape[0]/2)+1
    (x,y) = (x_,y_) = (0,0)
    for i in range(board[0][0].shape[0]):
        for j in range(board[0][0].shape[0]):
            if board[0][i][j]==1:
                (x,y) = (i/2,j+1)
            if board[1][i][j]==1:
                (x_,y_) = (i/2,j+1)

    blocks = board[2]+board[3]
    extraH = []
    print (" _" * n)    # header
    for row in range(n):
        rV = "| "*n + "|"
        rH = "-"*(2*n+1)
        blocks_row = row*2
        for col in range(n*2-1):
            if blocks[blocks_row][col]:
                rV = placePiece(rV, col+1,'x')
                if blocks_row < n*2-3 and blocks[blocks_row+2][col]:
                    extraH.append(col+1)
        if (x==row): rV = placePiece(rV, y, pieces[0])
        if (x_==row): rV = placePiece(rV, y_, pieces[1])
        if row != n-1:
            while (len(extraH)):
                next_extraH = extraH.pop()
                rH = placePiece(rH, next_extraH, 'x')
                            
            blocks_row = row*2 + 1
            for col in range(n*2-1):
                if blocks[blocks_row][col] and col+2 < n*2-1 and blocks[blocks_row][col+2]:
                    rH = placeHorizontalWall(rH, (col+1)/2, 'x')
        print (rV)
        print (rH)
