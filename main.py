from Coach import Coach
from quoridor.QuoridorGame import QuoridorGame as Game
from quoridor.pytorch.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 10, #1000
    'numEps': 50, #100
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp','5x5best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

if __name__=="__main__":
    g = Game(5)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
