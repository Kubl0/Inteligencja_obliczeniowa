import pandas as pd
from Methods import DecisionTree
from Methods import Preprocessing
from Methods import NaiveBayes_KN as NBK
from Methods import Neural
from sklearn.utils import shuffle

TRAIN_SIZE = 0.9


def main():
    dataD = Preprocessing.withDeletion(getData("./Data/winequality.csv"))
    dataM = Preprocessing.withMean(getData("./Data/winequality.csv"))

    # getResultsOfDT(dataD, dataM)
    # getResultsOfNB(dataD, dataM)
    # getResultsOfKNN(dataD, dataM)
    getResultsOfNeural(dataD, dataM)

    return 0


def getData(directory):
    data = pd.read_csv(directory)
    data = shuffle(data)
    data['quality'] = data['quality'].replace([3, 4, 5], 'low')
    data['quality'] = data['quality'].replace([6, 7], 'medium')
    data['quality'] = data['quality'].replace([8, 9], 'high')
    return data


def printResult(result):
    print("Accurarcy: ", result[0])
    # print("Confusion Matrix: ")
    # print(result[1])
    return ""


LAYERS = (10, 7)


def getResultsOfNeural(dataD, dataM):
    print("Without normalization:")
    print("Neural w/ Deletion: ")
    print(printResult(Neural.classify(dataD, TRAIN_SIZE, LAYERS)), end='')
    print("Neural w/ Mean: ")
    print(printResult(Neural.classify(dataM, TRAIN_SIZE, LAYERS)), end='')
    print("With normalization:")
    print("Neural w/ Deletion: ")
    print(printResult(Neural.classify(Preprocessing.normalization(dataD), TRAIN_SIZE, LAYERS)), end='')
    print("Neural w/ Mean: ")
    print(printResult(Neural.classify(Preprocessing.normalization(dataM), TRAIN_SIZE, LAYERS)), end='')


def getResultsOfDT(dataD, dataM):
    print("Without max-depth: ")
    print("Without normalization:")
    print("DT w/ Deletion: ")
    print(printResult(DecisionTree.classify(dataD, TRAIN_SIZE)), end='')
    print("DT w/ Mean: ")
    print(printResult(DecisionTree.classify(dataM, TRAIN_SIZE)), end='')
    print("With normalization:")
    print("DT w/ Deletion: ")
    print(printResult(DecisionTree.classify(Preprocessing.normalization(dataD), TRAIN_SIZE)), end='')
    print("DT w/ Mean: ")
    print(printResult(DecisionTree.classify(Preprocessing.normalization(dataM), TRAIN_SIZE)), end='')
    print("With max-depth: ")
    print("Without normalization:")
    print("DT w/ Deletion: ")
    print(printResult(DecisionTree.classifyMax(dataD, TRAIN_SIZE, 5)), end='')
    print("DT w/ Mean: ")
    print(printResult(DecisionTree.classifyMax(dataM, TRAIN_SIZE, 5)), end='')
    print("With normalization:")
    print("DT w/ Deletion: ")
    print(printResult(DecisionTree.classifyMax(Preprocessing.normalization(dataD), TRAIN_SIZE, 5)), end='')
    print("DT w/ Mean: ")
    print(printResult(DecisionTree.classifyMax(Preprocessing.normalization(dataM), TRAIN_SIZE, 5)), end='')


def getResultsOfNB(dataD, dataM):
    print("Without normalization:")
    print("NB w/ Deletion: ")
    print(printResult(NBK.classifyNB(dataD, TRAIN_SIZE)), end='')
    print("NB w/ Mean: ")
    print(printResult(NBK.classifyNB(dataM, TRAIN_SIZE)), end='')
    print("With normalization:")
    print("NB w/ Deletion: ")
    print(printResult(NBK.classifyNB(Preprocessing.normalization(dataD), TRAIN_SIZE)), end='')
    print("NB w/ Mean: ")
    print(printResult(NBK.classifyNB(Preprocessing.normalization(dataM), TRAIN_SIZE)), end='')


def getResultsOfKNN(dataD, dataM):
    print("Without normalization:")
    for i in range(8, 12):
        print("KNN w/ Deletion: ")
        print(printResult(NBK.classifyKN(dataD, TRAIN_SIZE, i)), end='')
        print("KNN w/ Mean: ")
        print(printResult(NBK.classifyKN(dataM, TRAIN_SIZE, i)), end='')
    print("With normalization:")
    for i in range(5, 10):
        print("KNN w/ Deletion: ")
        print(printResult(NBK.classifyKN(Preprocessing.normalization(dataD), TRAIN_SIZE, i)), end='')
        print("KNN w/ Mean: ")
        print(printResult(NBK.classifyKN(Preprocessing.normalization(dataM), TRAIN_SIZE, i)), end='')


if __name__ == "__main__":
    main()
