import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from preprocessor import pca
from preprocessor import split_train_test
from preprocessor import ConfusionMatrixHelper

# Neural Network Classifier using Stochastic Gradient Descent, ReLU is used as activation function, Cross Entropy error is used as loss evaluation metric.

class NeuralNetworkClassifier:

    # initialize two layers.
    layer_1 = {}
    layer_2 = {}

    def __init__(self, input_dimension, hidden_dimension, outputs):

        # initialize with random numbers
        self.layer_1['w'] = np.random.randn(hidden_dimension,input_dimension) / np.sqrt(input_dimension)
        self.layer_1['b'] = np.random.randn(hidden_dimension,1) / np.sqrt(hidden_dimension)
        self.layer_2['w'] = np.random.randn(outputs,hidden_dimension) / np.sqrt(hidden_dimension)
        self.layer_2['b'] = np.random.randn(outputs,1) / np.sqrt(hidden_dimension)
        self.input_dimension = input_dimension
        self.hidden_layer_dimension = hidden_dimension
        self.output_size = outputs
        self.test_images = np.array((1, 1))
        self.test_labels = np.array((1, 1))

    # ultility functions as static methods

    @staticmethod
    def ReLU(z):
        return np.array([i if i>0 else 0 for i in np.squeeze(z)])

    @staticmethod
    def DerivativeReLU(z):
        return np.array([1 if i>0 else 0 for i in np.squeeze(z)])

    @staticmethod
    def SoftMax(z):
        return np.exp(z) / sum(np.exp(z))

    @staticmethod
    def CrossEntropy(v, y):
        # implement the cross entropy error
        return -1.0 * np.log(v[y])
    
    # setters
    def SetTestData(self, x_test, y_test):
        self.test_images = x_test.copy()
        self.test_labels = y_test.copy()

    # class methods

    # calculate loss by feed-forward, return sum of loss for each data record
    def __Loss(self, x_train, y_train):
        loss = 0
        for n in range(len(x_train)):
            label = y_train[n]
            image = x_train[n]
            loss += self.__FeedForward(image, label)['cross_entropy_error']
        return loss

    def SimplyPredict(self, x_test):
        num_test = x_test.shape[0]
        predicted_labels = np.zeros(num_test)
        dummy_label = 0
        for i in range(num_test):
            image = x_test[i][:]
            predicted_labels[i] = np.argmax(self.__FeedForward(image, dummy_label)['predicted_label'])

        return predicted_labels

    # test accuracy based on given data and label
    def TestAccuracy(self, x_test, y_test):
        num_hit = 0
        num_test = x_test.shape[0]
        predicted_labels = np.zeros(num_test)

        for i in range(num_test):
            label = y_test[i]
            image = x_test[i][:]

            predicted_labels[i] = np.argmax(self.__FeedForward(image, label)['predicted_label'])

            if (predicted_labels[i] == label):
                num_hit += 1

            accuracy = num_hit / np.float(len(x_test))
        return predicted_labels, accuracy

    def __FeedForward(self, x, y):
        # After first layer..
        after_first_layer = np.matmul(self.layer_1['w'], x).reshape((self.hidden_layer_dimension, 1)) + self.layer_1['b']
        # After applied activation function
        act_func_applied = np.array(self.ReLU(after_first_layer)).reshape((self.hidden_layer_dimension, 1))
        # Raw output of the network
        raw_output = np.matmul(self.layer_2['w'], act_func_applied).reshape((self.output_size, 1)) + self.layer_2['b']
        # get predicted labels from Raw output
        predicted_y = np.squeeze(NeuralNetworkClassifier.SoftMax(raw_output))
        # calculate error
        cross_entropy_error = NeuralNetworkClassifier.CrossEntropy(predicted_y, y)
        
        result = { 'after_first_layer':after_first_layer,
                   'act_func_applied':act_func_applied,
                   'raw_output':raw_output,
                   'predicted_label':predicted_y.reshape((1,self.output_size)),
                   'cross_entropy_error':cross_entropy_error }
        return result

    def __BackPropagate(self, x, y, feed_forward_result):
        # implement the back propagation process, compute the gradients

        # one hot for result vector
        output_vector = np.zeros((1, self.output_size))
        output_vector[0][y] = 1

        # calculate gradient of layer #2
        dU = (-(output_vector - feed_forward_result['predicted_label'])).reshape((self.output_size,1))

        # graident for biases, simple deep copy
        gradient_layer_2_bias = np.copy(dU)
        # gradient for weights
        gradient_layer_2_weights = np.matmul(dU,feed_forward_result['act_func_applied'].transpose())

        # calculate gradient for layer #1
        delta = np.matmul(self.layer_2['w'].transpose(), dU)

        # gradient for layer 1 biases
        gradient_layer_1_bias = delta.reshape(self.hidden_layer_dimension, 1) * self.DerivativeReLU(feed_forward_result['after_first_layer']).reshape(self.hidden_layer_dimension, 1)
        # gradient for layer 1 weights
        gradient_layer_1_weights = np.matmul(gradient_layer_1_bias.reshape((self.hidden_layer_dimension, 1)),x.reshape((1, self.input_dimension)))

        # aggregate them and return
        gradients = { 'gradient_layer_2_weights':gradient_layer_2_weights,
                      'gradient_layer_2_bias':gradient_layer_2_bias,
                      'gradient_layer_1_bias':gradient_layer_1_bias,
                      'gradient_layer_1_weights':gradient_layer_1_weights }
        return gradients

    def __Update(self, back_propagate_values, learning_rate):
        # update the hyperparameters
        self.layer_1['w'] -= learning_rate * back_propagate_values['gradient_layer_1_weights']
        self.layer_1['b'] -= learning_rate * back_propagate_values['gradient_layer_1_bias']
        self.layer_2['w'] -= learning_rate * back_propagate_values['gradient_layer_2_weights']
        self.layer_2['b'] -= learning_rate * back_propagate_values['gradient_layer_2_bias']

    def Train(self, x_train, y_train, num_iterations = 1000, learning_rate = 0.5, test_accuracy = False):
        # Randomized indices for the training dataset
        rand_inds = np.random.choice(len(x_train), num_iterations)

        counters = []
        losses = []
        accuracies = []

        counter = 1
        for i in rand_inds:
            # feed forward the training data, back-propagate the result, and update the weights and biases.
            feed_forward_result = self.__FeedForward(x_train[i], y_train[i])
            back_propagate_result = self.__BackPropagate(x_train[i], y_train[i], feed_forward_result)
            self.__Update(back_propagate_result, learning_rate)
            
            if counter % 2000 == 0:
                print("Number of iterations run:", counter)
                if test_accuracy:
                    if counter % 20000 == 0:
                        print("Running accuracy test:")
                        loss = self.__Loss(x_train, y_train)
                        predicted_labels, accuracy = self.TestAccuracy(self.test_images, self.test_labels)
                        counters.append(counter)
                        losses.append(loss)
                        accuracies.append(accuracy)
                        print("Accuracy test: Accuracy:{:.2f}%, Loss:{:.2f}".format(accuracy * 100, loss))
            counter += 1

        print('Training finished!')
        return counters, losses, accuracies


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

# pca
pca_number_of_components = 100
_, train_images_recon = pca(train_images, pca_number_of_components)
_, valid_images_recon = pca(valid_images, pca_number_of_components, mean_source = train_images)
_, test_images_recon = pca(test_images, pca_number_of_components, mean_source = train_images)

# hyper parameters

# learning rate
LEARNING_RATE = 0.01
# number of iterations in training
NUM_ITERATIONS_TRAINING = 100000

# dimension of input, defined by number of input image pixels (28 * 28)
input_dimension = train_images_recon.shape[1]
# size of hidden layer
HIDDEN_DIMENSION = 300
# number of outputs, defined by number of output labels.
output_dimension = np.unique(train_labels).shape[0]

T_training_start = time.time()

# construct
neural = NeuralNetworkClassifier(input_dimension, HIDDEN_DIMENSION, output_dimension)
neural.SetTestData(valid_images, valid_labels)

# train the network
time_stamps, losses, accuracies = neural.Train(train_images_recon, train_labels, num_iterations=NUM_ITERATIONS_TRAINING, learning_rate=LEARNING_RATE, test_accuracy = True)

T_training_end = time.time()

# predict on validation, and draw confusion matrix
predicted, _ = neural.TestAccuracy(valid_images_recon, valid_labels)

ConfusionMatrixHelper.plot_confusion_mat(predicted.astype(np.int32), valid_labels, 10, "Neural Network")


# predict the 5000 set:
T_evaluation_start = time.time()

predicted_labels = neural.SimplyPredict(test_images_recon)

T_evaluation_end = time.time()

print("Neural Network train used: {} seconds".format(T_training_end - T_training_start))
print("Neural Network evaluation used: {} seconds".format(T_evaluation_end - T_evaluation_start))

output_file_name = './Output/predicted_labels_NeuralNetwork.h5'
with h5py.File(output_file_name,'w') as H:
    H.create_dataset('Output',data=predicted_labels)
    print("Neural Network: predicted labels saved as " + output_file_name)


## debug plots
#plt.plot(time_stamps, losses)
#plt.xlabel('Number of iterations')
#plt.ylabel('Loss')
#plt.title('Training Loss')
#plt.show()

#plt.plot(time_stamps, accuracies)
#plt.xlabel('Number of iterations')
#plt.ylabel('Accuracy')
#plt.title('Accuracy on test dataset')
#plt.show()
