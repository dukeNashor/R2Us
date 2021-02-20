import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import time

def split_train_test(images, labels, train_percent, random_seed = 0):
    np.random.seed(random_seed)
    images_labels = np.hstack((images, labels.reshape(len(labels), 1)))
    perm = np.random.permutation(len(images_labels))
    number_of_samples = len(images_labels)
    train_end = int(train_percent * number_of_samples)
    train = images_labels[perm[:train_end], :]
    test = images_labels[perm[train_end:], :]
    
    return train[:, 0 : -1].astype(np.float32), train[:, -1].astype(np.int32), test[:, 0 : -1].astype(np.float32), test[:, -1].astype(np.int32)

def pca(images, num_pc, mean_source = None):
    dim = images.shape[1]
    if num_pc > dim or num_pc < 1:
        num_pc = dim

    if mean_source is None:
        mean_source = images

    # de-mean the data:
    data_mean = images.mean(axis = 0)
    data = images - data_mean
    # get covariance matrix:
    mat_cov = np.cov(data.T) / data.shape[0]
    # eigen decomposition
    eig_values, eig_vecs = np.linalg.eig(mat_cov)
    # sort the eigenvectors in eigen-value descending order
    sort_idx = eig_values.argsort()[::-1]
    eig_values = eig_values[sort_idx]
    eig_vecs = eig_vecs[:, sort_idx]
    # get largest num_pc eigenvectors
    selected_pcs = eig_vecs[:, :num_pc]
    
    data_low_dim = data.dot(selected_pcs)
    
    data_reconstructed = data_low_dim.dot(selected_pcs.T) + data_mean
    
    ## debug code
    #plt.figure()
    #plt.imshow(data[0].reshape(28, 28), cmap=plt.get_cmap('gray'))
    #plt.show()
    #plt.imshow(data_reconstructed[0].reshape(28, 28),cmap=plt.get_cmap('gray'))
    #plt.show()

    return data_low_dim, data_reconstructed

class ConfusionMatrixHelper:

    # the x axis stands for predicted, y axis stands for truth
    @staticmethod
    def get_confusion_matrix(predicted, truth, number_of_labels):
        conf_mat = np.zeros((number_of_labels, number_of_labels), dtype=np.int32)
        num_records = predicted.size
        for i in range(num_records):
            conf_mat[truth[i]][predicted[i]] += 1
    
        return conf_mat

    @staticmethod
    def get_accuracy_by_confusion_matrix(conf_mat):
        return float(conf_mat.trace()) / np.sum(conf_mat)

    @staticmethod
    def plot_confusion_mat(predicted, truth, number_of_labels, title = ""):
        plt.figure(figsize=(10,7))
        conf_mat = ConfusionMatrixHelper.get_confusion_matrix(predicted, truth, number_of_labels)
        ax = sns.heatmap(conf_mat, annot=True, fmt="d")
        plt.ylabel('True')
        plt.xlabel('Predicted')

        accuracy = ConfusionMatrixHelper.get_accuracy_by_confusion_matrix(conf_mat)
        plt.title(title + "; Accuracy: {}".format(accuracy))
        plt.show()