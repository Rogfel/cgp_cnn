from sklearn.tree import DecisionTreeClassifier


def classification_model():
    return DecisionTreeClassifier(
        max_depth=10,  # Prevent overfitting
        min_samples_split=5,  # Minimum samples required to split
        min_samples_leaf=2,  # Minimum samples required at leaf node
        random_state=42
    )