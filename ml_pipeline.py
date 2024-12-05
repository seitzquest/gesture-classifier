import glob
import os
import pickle
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_handling import get_dataset

# Fix the random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create the project file structure
MODEL_DIR = "./models/"
DATASET_FILE = "./data/dataset.pkl"
MODEL_SCORES_FILE = "./model_scores.pkl"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def evaluate_and_store(clf, clf_id, X_test, y_test, metrics_df=False, criterion=None):
    """Evaluates the classifier on the test set and stores the model and score if it outperforms the previous one in terms of criterion

    Args:
        clf (object): Classifier that implements the method predict from BaseEstimator and ClassifierMixin of sklearn.base
        clf_id (string): Classifier id
        X_test (list[np.ndarray]): Test samples (3D time series of length NUM_FEATURES)
        y_test (list[int]): Test labels (class id between 1 to NUM_CLASSES)
        metrics_df (bool, optional): If True the classification_report is returned as a dataframe, otherwise dict. Defaults to False.
        criterion (callable, optional): Criterion for model comparison. Defaults to None (Weighted Avg F1 score).

    Returns:
        tuple[np.ndarray, dict | pd.DataFrame, float]: _description_
    """
    y_pred = clf.predict(X_test)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    score = metrics["weighted avg"]["f1-score"] if criterion is None else criterion(yt_test, y_pred)

    if os.path.exists(MODEL_SCORES_FILE):
        with open(MODEL_SCORES_FILE, "rb") as f:
            # PUBLIC_RESOURCE
            # Use joblib for np.array (e.g. models) and pickle for standard objects (e.g. metrics)
            # https://stackoverflow.com/questions/12615525/what-are-the-different-use-cases-of-joblib-versus-pickle
            model_scores = pickle.load(f)
            if clf_id not in get_available_pretrained_models() or score > model_scores[clf_id]:
                # Save the model if it outperforms the previous one (model versioning not implemented)
                model_path = os.path.join(MODEL_DIR, f"{clf_id}.pkl")
                joblib.dump(clf, model_path)

                model_scores[clf_id] = score
    else:
        model_scores = {clf_id: score}
        model_path = os.path.join(MODEL_DIR, f"{clf_id}.pkl")
        joblib.dump(clf, model_path)

    # Update model scores
    with open(MODEL_SCORES_FILE, "wb") as f:
        pickle.dump(model_scores, f)

    if metrics_df:
        metrics = pd.DataFrame(metrics)

    return y_pred, metrics, accuracy


def get_classifiers():
    """Returns the default classifiers

    You can add your own models by extending the dictionary.
    The pipeline assumes the models to implement the methods fit and predict from BaseEstimator and ClassifierMixin of sklearn.base.

    Returns:
        dict[string,object]: Mapping of classifier ids to untrained classifier instances
    """

    # PUBLIC_RESOURCE
    # Classifier selection was inspired by the following comparison:
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html  
    classifiers = {
        "logistic_regression": LogisticRegression(random_state=RANDOM_SEED),
        "random_forest_classifier": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "nearest_neighbors": KNeighborsClassifier(n_neighbors=3),
        "support_vector_machine": SVC(kernel='linear', random_state=RANDOM_SEED),
        "naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "ada_boost": AdaBoostClassifier(random_state=RANDOM_SEED), 
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "neural_network": MLPClassifier(random_state=RANDOM_SEED),
        "linear_discriminant_analysis": LinearDiscriminantAnalysis(),
    }
    return classifiers


def get_available_pretrained_models():
    """Returns all available pre-trained models

    Returns:
        dict[string,object]: Mapping of classifier ids to pre-trained classifier instances
    """
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.pkl"))
    return {os.path.splitext(os.path.basename(model_file))[0]: joblib.load(model_file) for model_file in model_files}


if __name__ == "__main__":
    # Load the data
    X_train, X_test, y_train, y_test = get_dataset()

    # Selected model based on weighted avg F1 score
    best_classifier = "neural_network"
    clf = get_classifiers()[best_classifier]
    clf.fit(X_train, y_train)

    # Basic model evaluation
    # A better overview is available on the evaluation app: streamlit run model_evaluation_app.py
    y_pred, metrics, accuracy = evaluate_and_store(clf, best_classifier, X_test, y_test, metrics_df=True)
    print(metrics.round(3))
    print(f"Accuracy: {accuracy:.2%}")