import Arena
from MCTS import MCTS
from quoridor.QuoridorGame import QuoridorGame, display
from quoridor.QuoridorPlayers import *
from quoridor.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = QuoridorGame(5)

# all players
#rp = RandomPlayer(g).play
#gp = GreedyOthelloPlayer(g).play
hp = HumanQuoridorPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/','checkpoint_7.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':0.7})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

n2 = NNet(g)
n2.load_checkpoint('./temp/','checkpoint_9.pth.tar')
args2 = dotdict({'numMCTSSims': 50, 'cpuct':0.7})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, n2p, g, display=display)
#arena = Arena.Arena(hp, n1p, g, display=display)

print(arena.playGames(40, verbose=True))
