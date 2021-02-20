import h5py
import numpy as np
import os

# load Naive Bayes result
predicted_labels_NB = np.array(1)
with h5py.File('./Output/predicted_labels_NaiveBayes.h5','r') as H:
    predicted_labels_NB = np.copy(H['Output']).astype(np.int32)
    print("predicted labels of Naive Bayes loaded.")

# load KNN result
predicted_labels_KNN = np.array(1)
with h5py.File('./Output/predicted_labels_KNN.h5','r') as H:
    predicted_labels_KNN = np.copy(H['Output']).astype(np.int32)
    print("predicted labels of KNN loaded.")

# load Neural Network result
predicted_labels_Neural = np.array(1)
with h5py.File('./Output/predicted_labels_NeuralNetwork.h5','r') as H:
    predicted_labels_Neural = np.copy(H['Output']).astype(np.int32)
    print("predicted labels of Neural Network loaded.")