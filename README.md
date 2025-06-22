# Software-Defect-Prediction

Software Defect Prediction is an
important aspect in order to ensure software
quality. Deep Learning techniques can also be
used for the same.

In this project we use Random forest, Convolutional Neural Networks,
SVM, Decision Tree, Naive Bayes and Artificial Neural Network to train the model with the data.
After getting different results from these techniques, we combine them through
Logistic Regression and get the final output.

We use different open source datasets from NASA
Promise Data Repository to perform this
comparative study.

For evaluation, three widely used metrics:
Accuracy, F1 scores and Areas under Receiver
Operating Characteristic curve are used. It is found
that Artificial Neural Network outperformed all the
other dimensionality reduction techniques.

# Detailed Summary

## Project Overview

| Item                | Description                                                                |
| ------------------- | -------------------------------------------------------------------------- |
| **Project Name**    | Software Defect Prediction using Ensemble Machine Learning & Deep Learning |
| **Main Goal**       | Predict defective software modules using multiple classifiers + ensemble   |
| **Primary Dataset** | `pc2.csv` from NASA Promise Data Repository                                |
| **Language**        | Python 3.x                                                                 |
| **Core Libraries**  | pandas, numpy, scikit-learn, keras, tensorflow, imbalanced-learn, seaborn  |

## Dataset Detail

| Attribute           | Description                          |
| ------------------- | ------------------------------------ |
| Source              | NASA Promise Repository              |
| File Used           | `pc2.csv`                            |
| Features            | Software code metrics                |
| Target Variable     | `defects` (1 = defective, 0 = clean) |
| Preprocessing Steps | Null check, SMOTE, train-test split  |

## Preprocessing Summary
| Step                  | Tool/Method                             |
| --------------------- | --------------------------------------- |
| Load data             | `pd.read_csv()`                         |
| Handle missing values | `.fillna(value=False)`                  |
| Train/Test split      | `train_test_split` (90%/10%)            |
| Balancing             | `SMOTE(sampling_strategy=1.0)`          |
| Further splitting     | Training → Train + Validation (90%/10%) |

## Model Architecture

| Model             | Library      | Key Parameters                                                      |
| ----------------- | ------------ | ------------------------------------------------------------------- |
| **ANN**           | Keras        | 15 → 8 → 5 → 1, ReLU+Sigmoid, 100 epochs, Adam, Binary Crossentropy |
| **CNN**           | Keras        | 3 Conv layers (64→32→16), Flatten, Dense, 40 epochs                 |
| **SVM**           | scikit-learn | Kernel: RBF (`gamma='auto'`)                                        |
| **Random Forest** | scikit-learn | 100 trees, Max depth: 5                                             |

## Ensemble Model : Logistic Regeression

| Component | Description                                       |
| --------- | ------------------------------------------------- |
| Inputs    | Outputs from SVM, RF, ANN, CNN on validation set  |
| Model     | Logistic Regression (from `sklearn.linear_model`) |
| Output    | Final prediction on test set                      |
| Benefit   | Combines strengths of all base classifiers        |

## Run Command
| Command                                                              | Purpose               |
| -------------------------------------------------------------------- | --------------------- |
| `git clone https://github.com/fantashy-ayush/Software-Defect-PM.git` | Clone the project     |
| `pip install -r requirements.txt`                                    | Install dependencies  |
| `python logistic_regression_ensembler.py`                            | Run the full pipeline |

