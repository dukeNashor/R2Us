import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from preprocessor import pca
from preprocessor import split_train_test
from preprocessor import ConfusionMatrixHelper

# KNN Classifier
class KNNClassifier:
    
    def __init__(self, train_images, train_labels, test_images, test_labels, valid_images, valid_labels):
        self.train_images = train_images.copy()
        self.train_labels = train_labels.copy()
        self.test_images = test_images.copy()
        self.test_labels = test_labels.copy()
        self.valid_images = valid_images.copy()
        self.valid_labels = valid_labels.copy()
        self.train_size = len(train_labels)
        self.valid_size = len(valid_labels)
        self.test_size = len(test_labels)

    @staticmethod
    def squared_euclidean_dist(x, y):
        return np.sum((x - y)**2)

    # assumes that the result is sorted in distance-ascending order
    def GetVoteResult(self, indices):
        result = self.train_labels[indices]
        # assign weights in bincount:
        weights = np.linspace(1.9, 1.0, len(result))
        most_frequent = np.argmax(np.bincount(result, weights = weights))
        # print("result: ", result, "most_frequent:", most_frequent)
        return most_frequent

    def Classify(self, data, K = 3):
        ## non-vectorized version:
        #dists = np.zeros(self.train_size)
        #for i in range(self.train_size):
        #    dists[i] = KNNClassifier.squared_euclidean_dist(data, self.train_images[i])

        # vectorized version (still requires outer loop):
        # dists = np.linalg.norm(np.tile(data.reshape(1, len(data)), (self.train_size, 1)) - self.train_images, axis=1)
        dists = np.linalg.norm(data - self.train_images, axis=1)

        sorted_indices = np.argsort(dists)
        selected = sorted_indices[:K]
        return self.GetVoteResult(selected)
        
    # test accuracy based on given data and label
    def TestAccuracy(self, test_data = None, test_label = None, K = 3):
        if test_data is None or test_label is None:
            test_data = self.valid_images
            test_label = self.valid_labels
        
        num_test = test_data.shape[0]
        num_train = self.train_size
        dists = np.zeros((num_test, num_train))
        # L2
        dists = (np.square(test_data)).sum(axis=1).reshape(num_test, 1) + (np.square(self.train_images)).sum(axis=1) \
           - 2 * test_data.dot(self.train_images.T)
        # L1
        #dists = np.abs(np.tile(np.sum(np.abs(test_data), axis=1).reshape(num_test, 1), (1, num_train)) - \
        #    np.tile(np.sum(self.train_images, axis=1).reshape(1, num_train), (num_test, 1)))

        # evaluate accuracy
        num_hit = 0

        predicted_labels = np.zeros(num_test, dtype=np.int32)

        for i in range(num_test):
            sorted_indices = np.argsort(dists[i, :], kind='mergesort')
            selected = sorted_indices[:K]
            predicted_labels[i] = self.GetVoteResult(selected)

            if predicted_labels[i] == test_label[i]:
                num_hit += 1

        accuracy = float(num_hit) / test_data.shape[0]
        return predicted_labels, accuracy

    def SimplyPredict(self, x_test, K = 3):
        num_test = x_test.shape[0]
        predicted_labels = np.zeros(num_test)

        dists = (np.square(x_test)).sum(axis=1).reshape(num_test, 1) + (np.square(self.train_images)).sum(axis=1) \
           - 2 * x_test.dot(self.train_images.T)

        for i in range(num_test):
            sorted_indices = np.argsort(dists[i, :], kind='mergesort')
            selected = sorted_indices[:K]
            predicted_labels[i] = self.GetVoteResult(selected)

        return predicted_labels

# wrapper for easy evaluation routine
def KNNBenchmark(train, train_label, valid, valid_label, test, test_label, k_values = None, use_pca = False, pca_number_of_components = 100):

    # prepare input
    # use raw input or pca? if pca, then how many components?
    train_images_recon = np.zeros((1, 1))
    test_images_recon = np.zeros((1, 1))
    valid_images_recon = np.zeros((1, 1))
    if use_pca:
        _, train_images_recon = pca(train_images, pca_number_of_components)
        _, test_images_recon = pca(test_images, pca_number_of_components, mean_source = train_images)
        _, valid_images_recon = pca(valid_images, pca_number_of_components, mean_source = train_images)
    else:
        train_images_recon = train.copy()
        test_images_recon = test.copy()
        valid_images_recon =valid.copy()

    # construct classifier
    knn = KNNClassifier(train_images_recon, train_label, test_images_recon, test_label, valid_images_recon, valid_label)

    if k_values is None:
        k_values = [ i + 1 for i in range(20)]

    best_k = 0
    best_accuracy = 0

    K_s = []
    accuracy_s = []

    for k in k_values:
        # use train data itself to do checking.
        # predicted_labels, accuracy = knn.VectorizedClassify(test_data = train_images[:1000, :], test_label = train_labels[:1000], K = k)

        predicted_labels, accuracy = knn.TestAccuracy(test_data = test_images, test_label = test_labels, K = k)
        K_s.append(k)
        accuracy_s.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
        print("Accuracy for K = {}: {}".format(k, accuracy))

    print("Best accuracy got for K = {}: {}".format(best_k, best_accuracy))

    #plt.plot(K_s, accuracy_s)
    #plt.title('Accuracy w.r.t K with raw input')
    #plt.xlabel('K')
    #plt.ylabel('Accuracy')
    #plt.show()

    # use best K to get confusion matrix

    return K_s, accuracy_s, predicted_labels


# load fashion mnist dataset
train_images_h5 = h5py.File("./Input/train/images_training.h5", 'r')
train_labels_h5 = h5py.File("./Input/train/labels_training.h5", 'r')
test_images_h5 = h5py.File("./Input/test/images_testing.h5", 'r')
test_labels_h5 = h5py.File("./Input/test/labels_testing_2000.h5", 'r')

# split the training into 70% train and 30% validation. and the final test is performed on the test dataset
train_images, train_labels, valid_images, valid_labels = \
split_train_test(train_images_h5['datatrain'][:].astype(np.float32), \
                 np.array(train_labels_h5['labeltrain'][:]).astype(np.int32), 0.7)

test_images = test_images_h5['datatest'][:].astype(np.float32)
test_labels = np.array(test_labels_h5['labeltest'][:]).astype(np.int32)

train_images_h5.close()
train_labels_h5.close()
test_images_h5.close()
test_labels_h5.close()

############## benchmark routine

## set up K values we gonna use in later tests
#k_values = [ i + 1 for i in range(20)]

## raw input
#ks_raw, accs_raw, _ = RunKNNEvaluation(train_images, train_labels, test_images, test_labels, valid_images, valid_labels, k_values, use_pca = False)
#plt.plot(ks_raw, accs_raw, label = "raw input")

## the number of principle components we are going to test:
#components = [50, 80, 200, 400, 600]

#for pc in components:
#    # pca reconstructed
#    ks_pca, accs_pca, _ = RunKNNEvaluation(train_images, train_labels, test_images, test_labels, valid_images, valid_labels, k_values, use_pca = True, pca_number_of_components = pc)
#    plt.plot(ks_pca, accs_pca, label = "pca: {} components".format(pc))

#plt.title('Accuracy w.r.t K with raw input')
#plt.xlabel('K')
#plt.ylabel('Accuracy')
#plt.legend()
#plt.show()

################# profile routine
#import time
#T1 = time.time()
#
## best hyperparameters determined by the benchmark routine
#k_values = [9]
#number_of_pc = 80
#
#ks, accs, predicted_labels = RunKNNEvaluation(train_images, train_labels, test_images, test_labels, valid_images, valid_labels, k_values, use_pca = True, pca_number_of_components = number_of_pc)
#
#T2 =time.time()
#print("KNN evaluation used: {} seconds".format(T2 - T1))

############### generation of output

k = 9
number_of_pc = 80
_, train_images_recon = pca(train_images, number_of_pc)
_, valid_images_recon = pca(valid_images, number_of_pc, mean_source = train_images)
_, test_images_recon = pca(test_images, number_of_pc, mean_source = train_images)

T1 = time.time()
knn = KNNClassifier(train_images_recon, train_labels, valid_images_recon, valid_labels, test_images_recon, test_labels)
predicted_labels = knn.SimplyPredict(test_images_recon, K = k)
T2 =time.time()
print("KNN evaluation used: {} seconds".format(T2 - T1))

# test on validation set
predicted, accuracy = knn.TestAccuracy(valid_images_recon, valid_labels)
ConfusionMatrixHelper.plot_confusion_mat(predicted, valid_labels, 10, "KNN")


output_file_name = './Output/predicted_labels_KNN.h5'
with h5py.File(output_file_name,'w') as H:
    H.create_dataset('Output',data=predicted_labels)
    print("KNN: predicted labels saved as " + output_file_name)

