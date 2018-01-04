import os
import random
import numpy as np
import tensorflow as tf
from dataset import Dataset
from utils import *
from tensorflow.contrib import rnn
from tensorflow.contrib import rnn


'''
Helper function to perform a linear transformation with possible non linear activation
'''
def nn_layer(data, weights, bias, activate_non_linearity):
    result = tf.add(tf.matmul(data, weights), bias)
    if activate_non_linearity:
        result = tf.nn.relu(result)
    return result

'''
Helper function to compute pearson correlation between to row vectos
'''
def pearson_correlation(x, y):
    mean_x, var_x = tf.nn.moments(x, [0])
    mean_y, var_y = tf.nn.moments(y, [0])
    std_x = tf.sqrt(var_x)
    std_y = tf.sqrt(var_y)
    mul_vec = tf.multiply((x - mean_x), (y - mean_y))
    covariance_x_y, _ = tf.nn.moments(mul_vec, [0])
    pearson = covariance_x_y / (std_x * std_y)
    return pearson


def BiRNN(x, n_hidden, lengths, reuse=None):
    lstm_fw_cell = rnn.GRUCell(n_hidden, reuse=reuse)
    lstm_bw_cell = rnn.GRUCell(n_hidden, reuse=reuse)
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32, sequence_length=lengths)
    return tf.concat((output_states[0], output_states[1]), axis=1)


class MBPredictor:

    '''
    Init all parameters of a convolution neural network (CNN) for protein-RNA binding prediction
    '''
    def __init__(self, data_paths, params, network_name, loggername):

        tf.reset_default_graph()
        # Store data paths
        self.seq_path = data_paths[0]
        self.struct_path = data_paths[1]

        # Store network parameters 
        self.params = params

        # Set fully connecetd layer data
        self.fc_size = 2 * params['rnn_cell_size']
        self.WFC1 = tf.Variable(tf.truncated_normal([self.fc_size, self.params['hidden_layer_size']], stddev=0.1))
        self.BFC1 = tf.Variable(tf.zeros(1))
        self.WFC2 = tf.Variable(tf.truncated_normal([self.params['hidden_layer_size'], 1], stddev=0.1))
        self.BFC2 = tf.Variable(tf.zeros(1))

        # Store loggin data
        self.loggername = loggername
        self.network_name = network_name


    '''
    Train a convolution neural network (CNN) for protein-RNA binding prediction
    '''
    def train(self):

        scope = "train/"
        #tf.reset_default_graph()

        # Get numpy arrays with data (training and testing)
        all_train_data, all_train_lengths, all_train_labels, all_train_size = read_combined_data(self.seq_path,    \
                                                                                                 self.struct_path, \
                                                                                                 self.params['max_seq_len'])
        # Create dataset object for batching
        dataset = Dataset(all_train_data, all_train_lengths, all_train_labels, self.params['batch_size'])

        # Define placeholders for data, lengths and labels
        rna_data = tf.placeholder(tf.float32, [None, self.params['max_seq_len'], self.params['structures'] + 4], name=scope+"rnadata")
        rna_lengths = tf.placeholder(tf.int32, [None], name=scope+"rnalengths")
        rna_labels = tf.placeholder(tf.float32, [None], name=scope+"rnalabels")

        # Run CNN network
        rnn_output = BiRNN(rna_data, self.params['rnn_cell_size'], rna_lengths)

        # First fully connected layer: from flat CNN output to hidden layer
        hidden_layer = nn_layer(rnn_output, self.WFC1, self.BFC1, True)

        # Second fully connected layer: from hidden layer to a single prediction value
        preds = nn_layer(hidden_layer, self.WFC2, self.BFC2, False)
        preds = tf.squeeze(preds, axis=1, name=scope+'rnapreds')

        # Regularization
        regularizer = tf.nn.l2_loss(self.WFC1) + tf.nn.l2_loss(self.BFC1) + tf.nn.l2_loss(self.WFC2) + tf.nn.l2_loss(self.BFC2)

        # Loss function
        loss_score = tf.nn.l2_loss(rna_labels - preds, name=scope+'rnaloss_score')
        #loss = loss_score
        loss = loss_score + 1.0 / float(self.params['batch_size']) * self.params['beta'] * regularizer

        train_step = tf.train.AdamOptimizer(self.params['lr'], name=scope+'rnastep').minimize(loss)

        # calc accuracy
        accuracy = tf.pow(pearson_correlation(preds, rna_labels), 1, name=scope+'rnaaccuracy')

        init = tf.global_variables_initializer()

        # Create a saver object to store optimal traininig network
        saver = tf.train.Saver()

        # Create a new session with the configuration graph
        with tf.Session() as sess:
                    
            # Start trainig by initializing the session
            sess.run(init)

            # Perform epochs till convergence (detected by overfitting)
            for epoch_index in range(self.params['num_epochs']):
               	batch_counter = 0
                while dataset.has_next_batch():
                            
                    batch_counter += 1

                    # Read next batch
                    rna_batch, lengths_batch, labels_batch = dataset.next_batch()
                    feed_dict = {rna_data: rna_batch, rna_lengths: lengths_batch, rna_labels: labels_batch}

                    # Perform batch optimization
                    sess.run(train_step, feed_dict=feed_dict)

                    # Output stats every once in a while
                    if batch_counter % self.params['stop_check_interval'] == 0:

                        # Compute batch loss and accuracy
                        train_loss, train_accuracy = sess.run([loss, accuracy], feed_dict=feed_dict)

                        # statistics
                        status = 'Train acc: {}\tTrain Loss: {}\t{}\t{}\t{}'.format(train_accuracy, train_loss, batch_counter, dataset.curr_ind, epoch_index)
                        print (status)

                        logger = open(self.loggername, "a")
                        logger.write("\t\t" + str(train_accuracy) + "\t" + str(dataset.curr_ind) + "\t" + str(epoch_index) + "\n")
                        logger.close()
 

                # Reset dataset for a possible next epoch            
                dataset.reset()

            save_path = saver.save(sess, self.network_name)


    '''
    Test trained network for protein-RNA binding prediction
    '''
    def test(self):

        test_data, test_lengths, test_labels, test_size = read_combined_data(self.seq_path, \
                                                                             self.struct_path, \
                                                                             self.params['max_seq_len'])

        # Create a test session
        tf.reset_default_graph()
        new_graph = tf.Graph()
        scope = "train/"
        with tf.Session(graph=new_graph) as test_sess:

            # Import the trained network
            network = tf.train.import_meta_graph(self.network_name + '.meta')
            # Load the trained parameters
            network.restore(test_sess, self.network_name)

            # Construct feed data using placeholder names
            rna_data = new_graph.get_tensor_by_name(scope+"rnadata:0")
            rna_lengths = new_graph.get_tensor_by_name(scope+"rnalengths:0")
            rna_labels = new_graph.get_tensor_by_name(scope+"rnalabels:0")
            feed_dict={rna_data: test_data, rna_lengths: test_lengths, rna_labels: test_labels}

            # Access the evaluation metric
            test_accuracy = new_graph.get_tensor_by_name(scope+"rnaaccuracy:0")
            #test_preds = new_graph.get_tensor_by_name("rnapreds:0")

            #result, preds = test_sess.run([test_accuracy, test_preds], feed_dict)
            result = test_sess.run(test_accuracy, feed_dict)

        # Remove network files
        os.remove(self.network_name + ".data-00000-of-00001")
        os.remove(self.network_name + ".index")
        os.remove(self.network_name + ".meta")

        return result
        #return result, preds
