from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# LightGBM を使った多値分類


def main():
    # Iris データセットを読み込む
    iris = datasets.load_iris()
    a = "abc"
    print(iris)
    X, y = iris.data, iris.target


main()
