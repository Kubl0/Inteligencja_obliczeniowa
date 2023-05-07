from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


def classifyKN(df, train_size, neighbors):
    train_set, test_set = train_test_split(df, train_size=train_size, random_state=36)

    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    classifier.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
    accurarcy = classifier.score(test_set.iloc[:, :-1], test_set.iloc[:, -1])
    predicted = classifier.predict(test_set.iloc[:, :-1])
    confusionMatrix = confusion_matrix(test_set.iloc[:, -1], predicted)
    return accurarcy, confusionMatrix


def classifyNB(df, train_size):
    train_set, test_set = train_test_split(df, train_size=train_size, random_state=36)

    classifier = GaussianNB()
    classifier.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
    accurarcy = classifier.score(test_set.iloc[:, :-1], test_set.iloc[:, -1])
    predicted = classifier.predict(test_set.iloc[:, :-1])
    confusionMatrix = confusion_matrix(test_set.iloc[:, -1], predicted)
    return accurarcy, confusionMatrix
