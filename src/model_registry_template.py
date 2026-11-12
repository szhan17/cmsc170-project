"""
Template for connecting the group's from-scratch classifiers to issue #13.

Copy this file to src/model_registry.py after the group has classifier files.
Do not leave placeholder functions in the active registry.

The experiment runner expects get_models() to return:
    {
        "KNN": callable,
        "Naive Bayes": callable,
        "Perceptron": callable,  # if included in the final group implementation
    }

Each callable receives:
    (X_train_scaled, y_train, X_validation_scaled)

and returns a 1-D array containing one 0/1 prediction per validation row.

Example adapter (replace imports and calls with your group's actual API):

    from knn import KNNClassifier

    def predict_knn(X_train, y_train, X_validation):
        model = KNNClassifier(k=5)
        model.fit(X_train, y_train)
        return model.predict(X_validation)

    def get_models():
        return {"KNN": predict_knn}

The adapter is intentionally not implemented yet because the group's
classifier source files have not been supplied.
"""


def get_models():
    """Return no classifiers until actual model adapters are added."""
    return {}
