# Gesture Classifier Pipeline 🤌
Comparison of various classification models on the uWave gesture recognition task.

Paper: [uWave: Accelerometer-based personalized gesture recognition and its applications](https://ieeexplore.ieee.org/document/4912759)\
Dataset: [uWaveGestureLibrary.zip](http://zhen-wang.appspot.com/rice/files/uwave/uWaveGestureLibrary.zip) (2.3 MB)

## Requirements
- [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 
- [unrar](https://www.rarlab.com/rar_add.htm)
- [unzip](https://linux.die.net/man/1/unzip) (or any other zip extraction tool)

## Installation
Create the environment and install the dependencies
```bash
conda env create --file=environment.yml
```

Download and extract the dataset to "data/" directory
```bash
wget -P data http://zhen-wang.appspot.com/rice/files/uwave/uWaveGestureLibrary.zip
unzip data/uWaveGestureLibrary.zip -d data
```



## Usage
### ML Pipeline
Train a neural network, run evaluation and print its classification report:
```python
from ml_pipeline import get_classifiers, train_classifier, evaluate_and_store
from data_handling import get_dataset

# Load the data
X_train, X_test, y_train, y_test = get_dataset()

# Selected model based on weighted avg F1 score
best_classifier = "neural_network"
clf = get_classifiers()[best_classifier]
train_classifier(clf, X_train, y_train)

# Basic model evaluation
# A better overview is available on the evaluation app: streamlit run model_evaluation_app.py
_, metrics, accuracy = evaluate_and_store(clf, best_classifier, X_test, y_test, metrics_df=True)
print(metrics.round(3))
print(f"Accuracy: {accuracy:.2%}")
```
```bash
                 1        2      3       4        5        6      7      8  accuracy  macro avg  weighted avg
precision    0.972    0.992    1.0    0.99    0.983    0.991    1.0    1.0     0.991      0.991         0.991
recall       0.972    0.985    1.0    0.99    1.000    0.983    1.0    1.0     0.991      0.991         0.991
f1-score     0.972    0.989    1.0    0.99    0.991    0.987    1.0    1.0     0.991      0.991         0.991
support    107.000  132.000  108.0  103.00  114.000  115.000  109.0  108.0     0.991    896.000       896.000
Accuracy: 99.11%
```

<b>Note:</b> ```get_classifiers()``` contains a dictionary of models with default parameters. You can add your own models by extending the dictionary. The pipeline assumes the models to implement the methods ```fit``` and ```predict``` from ```BaseEstimator``` and ```ClassifierMixin``` of ```sklearn.base```.


### Service
The ML pipeline can be accessed via a [FastAPI](https://github.com/tiangolo/fastapi) service
```bash
uvicorn gesture_classifier_api:app --reload
```

### Visual Evaluation
We provide a visual evaluation tool to compare the trained models' performance:
```bash
streamlit run model_evaluation_app.py
```

Additionally, we provide a notebook ```data_exploration.ipynb``` that was used for data exploration.
