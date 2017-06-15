from csv import reader
from random import shuffle


class TicTacToe:
    N_ROWS = 9
    N_OPTS = 3
    N_FOLDS = 5

    def __init__(self, filename):
        self.data = list()
        self.load(filename)
        shuffle(self.data)
        self.testLen = len(self.data) / self.N_FOLDS
        self.trainLen = (self.N_FOLDS - 1) * self.testLen

    def load(self, filename):
        with open(filename, 'r') as file:
            csvReader = reader(file)
            for row in csvReader:
                if not row:
                    continue
                self.data.append(self.formatData(row))

    def formatData(self, row):
        return [int(value.strip()) for value in row]

    def createTrainAndTestSets(self, foldIdx):
        self.trainSet = list()
        for i in range(self.N_FOLDS):
            fold = list(self.data[i*self.testLen:(i+1)*self.testLen])
            if i != foldIdx:
                for row in fold:
                    self.trainSet.append(row)
            else:
                self.testSet = fold
