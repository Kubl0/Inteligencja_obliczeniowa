from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def classify(df, train_size):
    train_set, test_set = train_test_split(df, train_size=train_size, random_state=36)
    classifier = DecisionTreeClassifier()
    classifier.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
    accurarcy = classifier.score(test_set.iloc[:, :-1], test_set.iloc[:, -1])
    predicted = classifier.predict(test_set.iloc[:, :-1])
    confusionMatrix = confusion_matrix(test_set.iloc[:, -1], predicted)
    return accurarcy, confusionMatrix


def classifyMax(df, train_size, max_depth):
    train_set, test_set = train_test_split(df, train_size=train_size, random_state=36)
    classifier = DecisionTreeClassifier(max_depth=max_depth)
    classifier.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
    accurarcy = classifier.score(test_set.iloc[:, :-1], test_set.iloc[:, -1])
    predicted = classifier.predict(test_set.iloc[:, :-1])
    confusionMatrix = confusion_matrix(test_set.iloc[:, -1], predicted)
    return accurarcy, confusionMatrix
