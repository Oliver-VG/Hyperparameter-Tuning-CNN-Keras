# Hyperparameter-Tuning-CNN-Keras
This repository demonstrates hyperparameter tuning of a Convolutional Neural Network (CNN) on the MNIST dataset using Keras Tuner.

Overview:\
- Objective: Use the HyperBand search algorithm in Keras Tuner to find optimal hyperparameters (like the number of convolutional filters, kernel size, dense layer size, dropout rate, and learning rate) for an MNIST digit classification task.
- Dataset: MNIST (handwritten digit images, 28×28 pixels).
- Frameworks: TensorFlow/Keras for the model definition and Keras Tuner for hyperparameter search.

Dependencies:\
Make sure you have the following packages installed (preferably in a virtual environment or conda environment):
- Python 3.7+
- TensorFlow (2.x)
- Keras Tuner
- NumPy
- Matplotlib
- scikit-learn (for the classification report and confusion matrix)

Example installation with pip: \
pip install \
tensorflow \
keras-tuner \
numpy \
matplotlib \
scikit-learn 

Usage: \
Clone this repository or download the .ipynb file: \
git clone https://github.com/Oliver-VG/Hyperparameter-Tuning-CNN-Keras.git \
cd Hyperparameter-Tuning-CNN-Keras

Install dependencies (as shown above or via a requirements.txt if provided).\
Run Jupyter Notebook or JupyterLab:\
jupyter notebook HPTuning-CNN-Keras.ipynb \
Then open the notebook in your browser. \
Execute the cell. Keras Tuner will run a hyperparameter search, saving logs to kt_dir/cnn_mnist/ by default. This can be customized in the tuner = kt.Hyperband(...) configuration. 

Notebook Details: \
- Imports
  TensorFlow, Keras Tuner, NumPy, Matplotlib, scikit-learn, etc.
- Data Loading & Preprocessing
  Loads the MNIST dataset from tensorflow.keras.datasets.
  Normalizes images to [0,1] range, expands dimensions to (batch, 28, 28, 1).
- Model Definition (HyperModel)
  Uses a build_model(hp) function to define search spaces for:
  Convolution filters (filters_1, filters_2)
  Kernel sizes (kernel_1, kernel_2)
  Dense layer units (dense_units)
  Dropout rate (dropout)
  Learning rate (learning_rate)
- Hyperparameter Tuning
  Utilizes HyperBand from Keras Tuner:
  objective='val_accuracy'
  max_epochs=30
  factor=3
  EarlyStopping callback to stop searching if validation loss fails to improve.
- Selecting the Best Model
  Extracts the best hyperparameters and rebuilds the model with them.
  (Optionally) Retrains the best model for more epochs.
- Evaluation & Visualization
  Evaluates test accuracy.
  Generates a classification report and confusion matrix.
  Plots the confusion matrix with Matplotlib.

Results: \
After the hyperparameter search, the notebook prints the best hyperparameters and test accuracy. \
You also see a classification report (precision, recall, F1-score) and a confusion matrix for MNIST digit classification. 

Sample output: \
Best Hyperparameters found:
  - filters_1: 32
  - kernel_1: 3
  - ...
Test Accuracy with Best Model: 0.9912
