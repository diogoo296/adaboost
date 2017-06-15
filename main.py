#!/usr/bin/env python
import sys
import getopt
import numpy as np
from tictactoe import TicTacToe
from adaboost import AdaBoost
from plot import plot


def getArgs(argv):
    nIterations = 0
    inputFile = ''
    outputFile = ''

    try:
        opts, args = getopt.getopt(argv, 'ht:i:o:')
    except getopt.GetoptError:
        print('Usage: python main.py <args>')
        print('-t <integer>: number of iterations')
        print('-i <string>: input file name')
        print('-o <string>: output file name')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Usage: python main.py <args>')
            print('-t <integer>: number of iterations')
            print('-i <string>: input file name')
            print('-o <string>: output file name')
            sys.exit()
        elif opt == "-t":
            nIterations = int(arg)
        elif opt == '-i':
            inputFile = arg
        elif opt == '-o':
            outputFile = arg

    return inputFile, outputFile, nIterations


def crossValidateAdaboost(inputFile, outputFile, nIterations):
    ticTacToe = TicTacToe(inputFile)
    avgEin = np.zeros(nIterations)
    avgEout = np.zeros(nIterations)

    for k in range(ticTacToe.N_FOLDS):
        ticTacToe.createTrainAndTestSets(k)
        adaboost = AdaBoost(ticTacToe)
        Ein, Eout = adaboost.train(ticTacToe, nIterations)
        avgEin = np.sum([avgEin, Ein], axis=0)
        avgEout = np.sum([avgEout, Eout], axis=0)
        print('--------------------------------------')

    return avgEin / ticTacToe.N_FOLDS, avgEout / ticTacToe.N_FOLDS


def main(argv):
    inputFile, outputFile, nIterations = getArgs(argv)
    Ein, Eout = crossValidateAdaboost(inputFile, outputFile, nIterations)
    plot(nIterations, Ein, Eout, outputFile)

if __name__ == "__main__":
    main(sys.argv[1:])
