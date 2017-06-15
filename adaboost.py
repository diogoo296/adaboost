#!/usr/bin/env python
import numpy as np


class DecisionStump:
    def __init__(self, idx, opt):
        self.idx = idx
        self.opt = opt


class AdaBoost:
    def __init__(self, ticTacToe):
        self.W = np.ones(ticTacToe.size) / ticTacToe.size
        self.alphas = list()
        self.choosenRows = [
            np.zeros(ticTacToe.N_OPTS) for i in range(ticTacToe.N_ROWS)
        ]
        self.stumps = list()

    def isCorrect(self, choice, realVal, y):
        return (choice == realVal and y == 1) or (choice != realVal and y == -1)

    def calcStumpsError(self, ticTacToe):
        stumps = [np.zeros(ticTacToe.N_OPTS) for i in range(ticTacToe.N_ROWS)]

        for i in range(ticTacToe.size):
            row = ticTacToe.data[i]
            for idx in range(ticTacToe.N_ROWS):
                for opt in range(ticTacToe.N_OPTS):
                    if (not self.isCorrect(opt, row[idx], row[-1])):
                        stumps[idx][opt] += self.W[i]

        return stumps

    def chooseBestStump(self, ticTacToe):
        stumpsErrors = self.calcStumpsError(ticTacToe)
        minError = 10
        minIdx = -1
        minOpt = -1

        for idx in range(ticTacToe.N_ROWS):
            for opt in range(ticTacToe.N_OPTS):
                choosen = self.choosenRows[idx][opt]
                if choosen == 0 and stumpsErrors[idx][opt] < minError:
                    minError = stumpsErrors[idx][opt]
                    minIdx = idx
                    minOpt = opt

        self.stumps.append(DecisionStump(minIdx, minOpt))
        self.choosenRows[minIdx][minOpt] = 1

        return minError

    def calcAlpha(self, error):
        self.alphas.append(0.5 * np.log((1 - error) / error))

    def predict(self, stump, row):
        return 1 if row[stump.idx] == stump.opt else -1

    def updateWeights(self, ticTacToe):
        for i in range(ticTacToe.size):
            row = ticTacToe.data[i]
            pred = self.predict(self.stumps[-1], row)
            self.W[i] *= np.exp(-self.alphas[-1] * pred * row[-1])

        self.W /= np.sum(self.W)

    def sign(self, value):
        return 1 if value >= 0 else -1

    def calcError(self, ticTacToe):
        error = 0
        for row in ticTacToe.data:
            total = 0
            for alpha, stump in zip(self.alphas, self.stumps):
                total += alpha * self.predict(stump, row)

            if self.sign(total) != row[-1]:
                error += 1

        return error / float(ticTacToe.size)

    def train(self, ticTacToe, nIterations):
        for i in range(nIterations):
            minError = self.chooseBestStump(ticTacToe)
            self.calcAlpha(minError)
            self.updateWeights(ticTacToe)

            error = self.calcError(ticTacToe)
            alpha = self.alphas[-1]
            stump = self.stumps[-1]
            print('t=%d, e=%.3f, a=%.3f, stump=(%d|%d)' % (
                i+1, error, alpha, stump.idx, stump.opt
            ))
