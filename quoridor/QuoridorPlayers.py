import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanQuoridorPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        #print valid

        while True:
            a_dic = {'u':0,'d':1,'l':2,'r':3,'ul':4,'dr':5,'dl':6,'ur':7}
            a = raw_input()
            if a in a_dic:
                a = a_dic[a]
                break
            b = raw_input()
            x, y = [int(x) for x in b.split(' ')]
            if (a=='h'):
                a = self.game.board.index_of_action(8, x*2-1, y*2)
            if (a=='v'):
                a = self.game.board.index_of_action(9, x*2, y*2+1)
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class GreedyQuoridorPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a
