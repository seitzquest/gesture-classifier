from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from data_handling import data_preprocessing, get_dataset
from ml_pipeline import (evaluate_and_store, get_available_pretrained_models,
                         get_classifiers)

# Define the API service
app = FastAPI(
    title="Celonis AI Machine Learning Challenge",
    description="This is a simple service that classifies hand gestures from a time series of",
    version="1.0.0"
)

class TrainRequestBody(BaseModel):
    """TrainingRequestBody object that contains the classifiers and the test set size.
    """
    classifiers: str = Field(..., title="Classifiers", example="random_forest_classifier,logistic_regression")
    test_set_size: float = Field(..., title="Test set size", example=0.2)

class TrainingInfo(BaseModel):
    """TrainingInfo object that contains the classifier name and the evaluation metrics.
    """
    classifier: str = Field(..., example="Random Forest Classifier", format="string")
    f1_score: float = Field(..., example=0.95, format="number")
    precision: float = Field(..., example=0.95, format="number")
    recall: float = Field(..., example=0.95, format="number")
    accuracy: float = Field(..., example=0.95, format="number")

class Prediction(BaseModel):
    """Prediction object that contains the classifier name and the predicted class.
    """
    classifier: str = Field(..., example="Random Forest Classifier", format="string")
    class_: int = Field(..., example=1, format="number") # underscore to avoid conflict with Python 


def evaluate_model(clf_id, clf, X_test, y_test):
    """Evaluates a classifier and stores the results.

    Args:
        clf_id (string): Name of the classifier.
        clf (object): Classifier object that implements the predict method from scikit-learn.
        X_test (list[np.ndarray]): Test set of 3D movement data in float64.
        y_test (list[int]): Test set of labels.

    Returns:
        TrainingInfo: TrainingInfo object that contains the classifier name and the evaluation metrics.
    """

    _, metrics, accuracy = evaluate_and_store(clf, clf_id, X_test, y_test)

    return TrainingInfo(
        classifier=clf_id,
        f1_score=metrics["weighted avg"]["f1-score"],
        precision=metrics["weighted avg"]["precision"],
        recall=metrics["weighted avg"]["recall"],
        accuracy=accuracy
    )

def predict_example(clf_id, clf, x):
    """Predicts the hand gesture from movement data.

    Args:
        classifier_name (string): Name of the classifier.
        clf (object): Classifier object that implements the predict method from scikit-learn.
        x (np.ndarray): NumPy array with the 3D movement data in float64.

    Returns:
        Prediction: Prediction object that contains the classifier name and the predicted class.
    """
    y_pred = clf.predict(x)[0]
    return Prediction(
        classifier=clf_id,
        class_=y_pred
    )


@app.post("/api/train", summary="Trains a set of classifiers", operation_id="train", 
          description="Starts the training of a set of classifiers to predict the hand gesture from the movement data.",
          responses={200: {"description": "Successful Response"}, 405: {"description": "Invalid Input"}})
def train_model(
    request: TrainRequestBody) -> List[TrainingInfo]:
    """API endpoint for training a set of classifiers with a specific train-test-split.

    Args:
        request (TrainRequestBody): Request body that contains the classifiers and the test set size.

    Raises:
        HTTPException: When the test set size is not in the range [0, 1].
        HTTPException: When any of the selected_classifiers is not specified in the classifiers dictionary.

    Returns:
        List[TrainingInfo]: List of training results for each classifier.
    """

    # Check input
    if request.test_set_size < 0 or request.test_set_size > 1:
        raise HTTPException(status_code=405, detail="Invalid input")
    
    selected_classifiers = request.classifiers.split(",")
    classifiers = get_classifiers()
    selected_models = {k: v for k, v in classifiers.items() if k in selected_classifiers}

    if len(selected_models) == len(selected_classifiers):
        X_train, X_test, y_train, y_test = get_dataset(test_size=request.test_set_size)

        for clf in selected_models.values():
            clf.fit(X_train, y_train)

        training_results = [evaluate_model(clf_id, clf, X_test, y_test) for clf_id, clf in selected_models.items()]
        return training_results
    else:
        raise HTTPException(status_code=405, detail="Invalid input")


@app.post("/api/predict", summary="Predicts the hand gesture from movement data.", operation_id="predict", 
          description="Consumes a movement dataset to predict the hand gesture class.",
          responses={200: {"description": "Successful prediction"}, 405: {"description": "A training of the model must be called before prediction."}})
def prediction(request: UploadFile = File(..., format="binary")) -> List[Prediction]:
    """API endpoint for predicting the hand gesture from movement data.

    Args:
        request (UploadFile, optional): Text file that contains 3D time series. Defaults to File(..., format="binary").

    Raises:
        HTTPException: When no pretrained model is available for prediction.

    Returns:
        List[Prediction]: List of predictions for each pretrained model.
    """

    x = [np.loadtxt(request.file)]
    # PUBLIC_RESOURCE
    # X_train is used for fitting the standard scaler
    # https://datascience.stackexchange.com/questions/27615/should-we-apply-normalization-to-test-data-as-well
    X_train, _, _, _ = get_dataset()
    _, x = data_preprocessing(X_train, x)
    
    # Load the models
    available_models = get_available_pretrained_models()
    if not available_models:
        raise HTTPException(status_code=405, detail="A training of the model must be called before prediction")

    predictions = [predict_example(clf_id, clf, x) for clf_id, clf in available_models.items()]
    return predictions

if __name__ == "__main__":
    # Start app: uvicorn gesture_classifier_api:app --reload
    # API Documentation: http://127.0.0.1:8000/docs
    uvicorn.run(app, host="0.0.0.0", port=8000)