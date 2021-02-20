import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from scipy.stats import multivariate_normal

from preprocessor import pca
from preprocessor import split_train_test
from preprocessor import ConfusionMatrixHelper

# naive bayes classifier
class NaiveBayesClassifier:

    def __init__(self, train_images, train_labels):
        self.train_images = train_images.copy()
        self.train_labels = train_labels.copy()
        self.prior_prob = {}
        # placeholders
        self.unique_labels = np.array(1)
        self.gaussian_means = np.array(1)
        self.gaussian_variances = np.array(1)
        # cache for easy access to numeric attributes
        self.num_labels = 0
        self.input_dimension = 0

    def Train(self):
        # get unique labels and occurences
        self.unique_labels, counts = np.unique(train_labels, return_counts = True)
        self.num_labels = len(self.unique_labels)
        self.input_dimension = self.train_images[0].size
        # use proportion of each label as the prior probability:
        self.prior_prob = dict(zip(self.unique_labels, counts))
        # resize the gaussian parameter arrays
        # we use the array index as label. since we know in prior that the dataset labels range from 0 to 9
        # I know it is bad practice since labels can be not continous integers starting from zero,
        # but this is only used to avoid using Python's dict since dict uses hashing
        # and could cause the runtime performance to drop
        self.gaussian_means = np.zeros((self.num_labels, self.input_dimension), dtype=np.float32)
        self.gaussian_variances = np.zeros((self.num_labels, self.input_dimension), dtype=np.float32)

        # NOTICE: the offset in variance is applied to avoid sigular matrix in the following step.
        # reference: https://stackoverflow.com/questions/35273908/scipy-stats-multivariate-normal-raising-linalgerror-singular-matrix-even-thou
        OFFSET_VAR = 10e-10
        # get images of the same labels, calculate their mean and variance, register into the gaussian params.
        for l in self.unique_labels:
            current_label_images = self.train_images[self.train_labels == l, :]
            mean = current_label_images.mean(axis = 0)
            variance = current_label_images.var(axis = 0)
            self.gaussian_means[l][:] = mean
            self.gaussian_variances[l][:] = variance + OFFSET_VAR

        ## this produces Fig. 1.1 in our report
        ## debug code to show the model for each label
        #for i in range(self.num_labels):
        #    plt.subplot(1, self.num_labels, i + 1)
        #    plt.axis('off')
        #    # another hard coded part which should be noticed. since 
        #    # it's debug code then I guess we are OK with that.
        #    plt.title(str(i))
        #    plt.imshow(self.gaussian_means[i][:].reshape(28, 28))
        #plt.show()

    def SimplyPredict(self, test_images):
        num_test = test_images.shape[0]
        # matrix to store probability for each test image
        prob = np.zeros((num_test, self.num_labels))

        for i in self.unique_labels:
            mean = self.gaussian_means[i, :]
            variance = self.gaussian_variances[i, :]
            # calculate probability for current label;
            # NOTICE: the multivariate_normal checks whether any of the eigenvalues of the covariance matrix are less than some tolerance
            prob[:, i] = multivariate_normal.logpdf(test_images, mean = mean, cov = variance) + np.log(self.prior_prob[i])

        # return the index of the largest element, which corresponds to the most likely label
        return np.argmax(prob, axis = 1)

    # test accuracy based on given data and label
    def TestAccuracy(self, test_images, test_labels):
        # perform prediction
        predicted_labels = self.SimplyPredict(test_images)
        
        # compare with ground truth
        num_accurate = np.count_nonzero(predicted_labels == test_labels)

        accuracy = float(num_accurate) / test_images.shape[0]
        return predicted_labels, accuracy


def NaiveBayesBenchmark(train, train_label, valid, valid_label, test, test_label, use_pca = False, pca_number_of_components = 100):

    train_images_recon = np.array(1)
    valid_images_recon = np.array(1)
    test_images_recon = np.array(1)

    print("Naive Bayes: Bench marking with use_pca = {}, number_of_components = {}".format(use_pca, pca_number_of_components))
    input_dimension = train.shape[1]
    if use_pca:
        if pca_number_of_components < 0 or pca_number_of_components > input_dimension:
            pca_nupca_number_of_components = input_dimension

        # preprocess PCA
        _, train_images_recon = pca(train_images, pca_number_of_components)
        _, valid_images_recon = pca(valid_images, pca_number_of_components, mean_source = train_images)
        _, test_images_recon = pca(test_images, pca_number_of_components, mean_source = train_images)
    else:
        train_images_recon = train_images.copy()
        valid_images_recon = valid_images.copy()
        test_images_recon = test_images.copy()

    nbc = NaiveBayesClassifier(train_images_recon, train_labels)
    nbc.Train()
    predicted_labels, accuracy = nbc.TestAccuracy(valid_images_recon, valid_labels)
    print("Naive Bayes accuracy on validation set: {}%".format(accuracy * 100))
    return accuracy



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


################ benchmark part
## coarse step
#number_of_pc_s = range(10, 784, 10)
## fine step
#number_of_pc_s = range(10, 100, 2)
#accuracy_s = []
#
#for n in number_of_pc_s:
#    accuracy = NaiveBayesBenchmark(train_images, train_labels, valid_images, valid_labels, test_images, test_labels, use_pca = True, pca_number_of_components = n)
#    accuracy_s.append(accuracy)
#
#
#plt.plot(number_of_pc_s, accuracy_s)
#plt.xlabel('Accuracy on validation set')
#plt.ylabel('Number of principle components chosen')
#plt.title('Accuracy w.r.t. the number of PCA components')
#plt.show()
#
##############################################


number_of_pc = 22
# preprocess PCA
_, train_images_recon = pca(train_images, number_of_pc)
_, valid_images_recon = pca(valid_images, number_of_pc, mean_source = train_images)
_, test_images_recon = pca(test_images, number_of_pc, mean_source = train_images)

nbc = NaiveBayesClassifier(train_images_recon, train_labels)

T_training_start = time.time()

# train the classifier
nbc.Train()

T_training_end = time.time()

print("Naive Bayes train used: {} seconds".format(T_training_end - T_training_start))

T_test_start = time.time()

# test on validation set
predicted, accuracy = nbc.TestAccuracy(valid_images_recon, valid_labels)
print("Naive Bayes accuracy on validation set: {}%".format(accuracy * 100))
T_test_end = time.time()

print("Naive Bayes test on validation set used: {} seconds".format(T_test_end - T_test_start))

ConfusionMatrixHelper.plot_confusion_mat(predicted, valid_labels, 10, "Naive Bayes")

# generate h5 for test 5000
predicted_labels = nbc.SimplyPredict(test_images_recon)

output_file_name = './Output/predicted_labels_NaiveBayes.h5'
with h5py.File(output_file_name,'w') as H:
    H.create_dataset('Output',data=predicted_labels)
    print("Neural Network: predicted labels saved as " + output_file_name)