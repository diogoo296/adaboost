# AdaBoost
An implementation of Adaboost for an university assignment (Machine Learning - Universidade Federal de Minas Gerais).

## Introduction
[Adaptive Boosting](https://en.wikipedia.org/wiki/AdaBoost) is a Boosting learning algorithm which combines simple models (weak learners) in order to build a strong learner sensitive to noisy data and outliers and less prone to [overfitting](https://en.wikipedia.org/wiki/Overfitting).

## Dependencies

* Python (2.7)
* [NumPy](http://www.numpy.org/)
* [Matplotlib](https://matplotlib.org/)

## How to Run

The `main.py` file accepts the following command line arguments:
* **-t:** number of iterations (max: 27).
* **-i:** input file.
* **-o:** output file (errors line plot).
* **-h:** shows possible command line arguments.

Example of execution command:
```
./main.py -t 27 -i data_formated.csv -o errors.out
```

## Dataset
This implementation was adapted to classify TicTacToe games from the dataset downloaded at the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame). This data is composed by 958 instances where *x* is assumed to have played first and where the class - *positive* or *negative* - represents if the *x* player won or lose, respectively.

The original dataset (`data.csv`) was formatted using the commands listed below in the text editor [Vim](http://www.vim.org/), resulting in the file `data_formated.csv`:
```
:%s/positive/1/g
:%s/negative/-1/g
:%s/b/0/g
:%s/x/1/g
:%s/o/2/g
```

## Results
As shown by the image below, the test error - E(out) - follows the training error - E(in) - by decreasing when the second one goes down and by increasing when the training error goes up, as it should be expected in a AdaBoost algorithm.

![Results](results.png)

The average accuracy was about **76.4%**.
