#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cycler import cycler


def plot(nIterations, Ein, Eout, outputFile):
    plt.gca().set_prop_cycle(cycler('color', ['b', 'g']))
    fig = plt.figure()
    plt.plot(np.arange(0, nIterations), Ein, label='E(in)')
    plt.plot(np.arange(0, nIterations), Eout, label='E(out)')
    fig.suptitle('Adaboost error avg with cross-validation (k=5)')
    plt.xlabel("# Iteration")
    plt.ylabel("Error")
    plt.legend(['E(in)', 'E(out)'], loc='upper right')
    plt.show()
    pp = PdfPages(outputFile)
    pp.savefig(fig)
    pp.close()
