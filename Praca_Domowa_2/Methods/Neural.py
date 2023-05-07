from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def classify(df, train_size, layers):
    train_set, test_set = train_test_split(df, train_size=train_size, random_state=2137)

    # param_grid = {
    #     'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 50, 25), (10, 20, 30, 40), (100,50,100)],
    #     'activation': ['relu', 'tanh'],
    #     'solver': ['adam', 'sgd'],
    # }
    #
    # classifier = MLPClassifier()
    # grid_search = GridSearchCV(classifier, param_grid, cv=5)
    # grid_search.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)

    classifier = MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='tanh', solver='adam', max_iter=2000)
    classifier.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])

    predicted = classifier.predict(test_set.iloc[:, :-1])
    confusionMatrix = confusion_matrix(test_set.iloc[:, -1], predicted)
    accurarcy = accuracy_score(test_set.iloc[:, -1], predicted)
    return accurarcy, confusionMatrix
