#!/usr/bin/env python
import numpy as np


class DecisionStump:
    def __init__(self, idx, opt):
        self.idx = idx
        self.opt = opt


class AdaBoost:
    def __init__(self, tictactoe):
        self.W = np.ones(tictactoe.size) / tictactoe.size
        self.alphas = list()
        self.choosenRows = [
            np.zeros(tictactoe.nOpts) for i in range(tictactoe.rowSize)
        ]
        self.stumps = list()

    def sign(self, x1, x2, y):
        if (x1 == x2 and y == -1) or (x1 != x2 and y == 1):
            return -1
        else:
            return 1

    def calcStumpsError(self, ticTacToe):
        stumps = [np.zeros(ticTacToe.nOpts) for i in range(ticTacToe.rowSize)]

        for i in range(ticTacToe.size):
            row = ticTacToe.data[i]
            for idx in range(ticTacToe.rowSize):
                for opt in range(ticTacToe.nOpts):
                    if (self.sign(row[idx], opt, row[-1]) == -1):
                        stumps[idx][opt] += self.W[i]

        return stumps

    def findBestStump(self, stumps, nRows, nOpts):
        minStump = 9999999
        minIdx = -1
        minOpt = -1

        for idx in range(nRows):
            for opt in range(nOpts):
                choosen = self.choosenRows[idx][opt]
                if choosen == 0 and stumps[idx][opt] < minStump:
                    minStump = stumps[idx][opt]
                    minIdx = idx
                    minOpt = opt

        return minIdx, minOpt

    def calcAlpha(self, error):
        self.alphas.append(0.5 * np.log((1 - error) / error))

    def updateWeights(self, tictactoe, idx, opt):
        for i in range(tictactoe.size):
            row = tictactoe.data[i]
            h_row = 1
            if row[idx] != opt:
                h_row = -1
            self.W[i] = self.W[i] * np.exp(-self.alphas[-1] * h_row * row[-1])

        self.W /= np.sum(self.W)

    def calcError(self, tictactoe):
        error = 0
        for row in tictactoe.data:
            total = 0
            for i in range(len(self.stumps)):
                s = self.stumps[i]
                h_row = 1
                if row[s.idx] != s.opt:
                    h_row = -1
                total += self.alphas[i] * h_row

            if (total < 0 and row[-1] == 1) or (total >= 0 and row[-1] == -1):
                error += 1

        return error / float(tictactoe.size)

    def train(self, tictactoe, nIterations):
        for i in range(nIterations):
            stumps = self.calcStumpsError(tictactoe)
            rowSize = tictactoe.rowSize
            idx, opt = self.findBestStump(stumps, rowSize, tictactoe.nOpts)
            self.choosenRows[idx][opt] = 1
            self.calcAlpha(stumps[idx][opt])
            self.stumps.append(DecisionStump(idx, opt))
            self.updateWeights(tictactoe, idx, opt)

            error = self.calcError(tictactoe)
            print('t=%d, error=%.3f' % (i+1, error))
            alpha = self.alphas[-1]
            stump = self.stumps[-1]
            print(
                'a=%.3f, stump: idx=%d,opt=%d' % (alpha, stump.idx, stump.opt)
            )
