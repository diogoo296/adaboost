#!/usr/bin/env python
from csv import reader


class TicTacToe:
    N_ROWS = 9
    N_OPTS = 3
    N_FOLDS = 5

    def __init__(self, filename):
        self.data = list()
        self.load(filename)
        self.size = len(self.data)
        self.foldLen = self.size / self.N_FOLDS

    def load(self, filename):
        with open(filename, 'r') as file:
            csvReader = reader(file)
            for row in csvReader:
                if not row:
                    continue
                self.data.append(self.formatData(row))

    def formatData(self, row):
        return [int(value.strip()) for value in row]
