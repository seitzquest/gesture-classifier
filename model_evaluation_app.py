
import glob
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, log_loss)

from data_handling import NUM_CLASSES, get_dataset
from ml_pipeline import MODEL_DIR

# Set page title and favicon
st.set_page_config(page_title="Gesture Classifier Comparison", page_icon="🖐")

# Set the style of the plots for darkmode
params = {
          "ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w",
          "savefig.transparent" : True,
        }
plt.rcParams.update(params)

# Number of confusion matrices to display per row
NUM_CONFUSION_MATRICES_PER_ROW = 3

def id_to_name(id):
    """Converts a classifier id (snake_case) to its name.

    Args:
        id (string): Id of the classifier.

    Returns:
        string: Name of the classifier.
    """
    return id.replace("_", " ").title()

def name_to_id(name):
    """Converts a classifier name to its id (snake_case).

    Args:
        name (string): Name of the classifier.

    Returns:
        string: Id of the classifier.
    """
    return name.lower().replace(" ", "_")

@st.cache_data
def prediction(clf_id, _clf, _X_test):
    """Predicts the class predictions for the test set.
       Caches the result for a clf_id to avoid re-computation. If a model is updated, restart the app.

    Args:
        clf_id (string): Id of the classifier. Used to cache the result.
        _clf (object): Classifier object that implements the predict_proba method from scikit-learn.
        _X_test (list[np.ndarray]): Test set of 3D movement data in float64.

    Returns:
        np.ndarray: Predicted class for each time series in the test set.
    """
    y_pred = _clf.predict(_X_test)
    return y_pred

@st.cache_data
def prediction_proba(clf_id, _clf, _X_test):
    """Predicts the class probabilities for the test set.
       Caches the result for a clf_id to avoid re-computation. If a model is updated, restart the app.

    Args:
        clf_id (string): Id of the classifier. Used to cache the result.
        _clf (object): Classifier object that implements the predict_proba method from scikit-learn.
        _X_test (list[np.ndarray]): Test set of 3D movement data in float64.

    Returns:
        np.ndarray: Class probabilities for the test set.
    """
    proba = _clf.predict_proba(_X_test)
    return proba


def plot_confusion_matrix(ax, matrix, model_name):
    """Plot a confusion matrix.

    Args:
        ax (plt.axes._axes.Axes): Axis to plot the confusion matrix.
        matrix (np.ndarray): Confusion matrix.
        model_name (string): Classifier name.
    """
    classes = np.arange(1,NUM_CLASSES+1)
    cmap = sns.light_palette("#5CFE50", as_cmap=True, n_colors=len(classes))

    # Make zero cells white to highlight the errors
    off_zero_cells = matrix == 0
    sns.heatmap(matrix, ax=ax, mask=off_zero_cells, cmap=cmap, fmt="d", annot=True, cbar=False, xticklabels=classes, yticklabels=classes)

    sns.heatmap(matrix, ax=ax, mask=~off_zero_cells, cmap=["#FFFFFF"], fmt="d", annot=True, cbar=False, xticklabels=classes, yticklabels=classes)

    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    ax.set_title(f"{model_name}")
    ax.title.set_color('w')
    plt.yticks(rotation=0)


def main():
    """Implements the streamlit frontend for the gesture classifier comparison.

    Raises:
        Exception: If no model files are found in the model directory.
    """

    with open( "style.css" ) as css:
        st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

        st.title("Gesture Classifier Comparison 🤌")

        # Load the pre-trained models
        model_files = glob.glob(os.path.join(MODEL_DIR, "*.pkl"))
        if not model_files:
            raise Exception("A training of the model must be called before prediction")
        
        _, X_test, _, y_test = get_dataset()

        available_models = {os.path.splitext(os.path.basename(model_file))[0]: joblib.load(model_file) for model_file in model_files}
        
        st.markdown("#")
        selected_model_names = st.multiselect("Selected Models:", [id_to_name(x) for x in available_models.keys()], default=["Random Forest Classifier", "Logistic Regression"])    
        selected_models = [name_to_id(x) for x in selected_model_names]

        if selected_models:
            predictions = {clf_id: prediction(clf_id, clf, X_test) for clf_id, clf in available_models.items() if clf_id in selected_models} 
            probabilities = {clf_id: prediction_proba(clf_id, clf, X_test) for clf_id, clf in available_models.items() if clf_id in selected_models and hasattr(clf, 'predict_proba')} 

            st.markdown("#")

            st.subheader("Scores")
            st.write("Weighted average classification metric scores for each classifier model.")
            scores = pd.DataFrame()   
            for i, clf_id in enumerate(selected_models):
                y_pred = predictions[clf_id]
                report = classification_report(y_test, y_pred, output_dict=True)

                scores[id_to_name(clf_id)] = pd.DataFrame(report)["weighted avg"].T
                scores.loc["accuracy", id_to_name(clf_id)] = accuracy_score(y_test, y_pred)

            scores.drop("support", inplace=True)

            for clf_id, proba in probabilities.items():
                scores.loc["neg cross entropy", id_to_name(clf_id)] = -log_loss(y_test, proba)

            scores = scores.T.sort_values(by=["f1-score"], ascending=False)
            st.dataframe(scores.style.highlight_max(props="color: #000000; background-color: #5CFE50; border-color: #5CFE50;").format("{:.1%}"))
            st.markdown("#")

            st.subheader("Confusion Matrices")   
            st.write("Confusion matrices for each classifier model.")
            
            cols = [st.columns(NUM_CONFUSION_MATRICES_PER_ROW) for _ in range(len(available_models)//NUM_CONFUSION_MATRICES_PER_ROW + 1)]

            for i, clf_id in enumerate(selected_models):
                y_pred = predictions[clf_id]
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(5, 5))
                plot_confusion_matrix(ax, cm, id_to_name(clf_id))
                cols[i//NUM_CONFUSION_MATRICES_PER_ROW][i%NUM_CONFUSION_MATRICES_PER_ROW].pyplot(fig)


        multi_css=f'''
        <style>
        .stMultiSelect div div div div div:nth-of-type(2) {{visibility: hidden;}}
        .stMultiSelect div div div div div:nth-of-type(2)::before {{visibility: visible; content:"{"Choose a classifier"}"; color: #ffffff;}}
        </style>
        '''
        st.markdown(multi_css, unsafe_allow_html=True)

if __name__ == "__main__":
    main()