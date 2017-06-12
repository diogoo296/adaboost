#!/usr/bin/env python
import sys
import getopt
from tictactoe import TicTacToe
from adaboost import AdaBoost


def main(argv):
    nIterations = 0
    inputFile = ''
    # outputFile = ''

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
        # elif opt == '-o':
        #     outputFile = arg

    tictactoe = TicTacToe(inputFile)
    adaboost = AdaBoost(tictactoe)
    adaboost.train(tictactoe, nIterations)

if __name__ == "__main__":
    main(sys.argv[1:])
