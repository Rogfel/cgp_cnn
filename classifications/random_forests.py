from sklearn.ensemble import RandomForestClassifier


def classification_model(n_estimators=10):
    return RandomForestClassifier(n_estimators=n_estimators)


if __name__ == '__main__':
    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    RF = classification_model()
    RF.fit(X, Y)
    print(RF.predict([[2., 2.]]))


