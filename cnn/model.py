import os
import numpy as np
import tensorflow as tf
from dataset import Dataset
from utils import *
from cnn_utils import *
from tensorflow.contrib import rnn

# Debug object
zeros = tf.zeros((1, 6), dtype=tf.float32)

class CGraph:

    def __init__(self, predictor, hyper_index, scope):
        self.scope = scope
        self.predictor = predictor
        self.hyper_index = hyper_index
        self.rna_data = None
        self.rna_lengths = None
        self.rna_labels = None
        self.conv_output = None
        self.fcl_input = None
        self.hidden_layer = None
        self.preds = None
        self.regularizer = None
        self.loss_score = None
        self.loss = None
        self.train_step = None
        self.accuracy = None


    def build(self):

        # Get current filters
        p = self.predictor
        filters = p.filters_set[self.hyper_index]

        # Re-init the graph using the hyperparameters
        p.init_hyperparams(self.hyper_index)

        # Define placeholders for data, lengths and labels
        self.rna_data = tf.placeholder(tf.float32, [None, p.params['max_seq_len'], p.params['structures'], 2], name=self.scope+"rnadata")
        self.rna_lengths = tf.placeholder(tf.int32, [None], name=self.scope+"rnalengths")
        self.rna_labels = tf.placeholder(tf.float32, [None], name=self.scope+"rnalabels")

        # Define the CNN network
        self.conv_output = create_multiple_filter_cnn(self.rna_data, p.layer, filters, p.weights, p.biases, p.strides_x_y, p.kpool_x_y)

        # Flatten the CNN output for the fully connected layer input
        self.fcl_input = flatten(self.conv_output, p.fc_size)

        # First fully connected layer: from flat CNN output to hidden layer
        self.hidden_layer = nn_layer(self.fcl_input, p.WFC1, p.BFC1, True)

        # Second fully connected layer: from hidden layer to a single prediction value
        self.preds = nn_layer(self.hidden_layer, p.WFC2, p.BFC2, False)
        self.preds = tf.squeeze(self.preds, axis=1, name=self.scope+'rnapreds')

        # Regularization - ADD MORE ARGS
        #self.regularizer = tf.nn.l2_loss(p.WFC1) + tf.nn.l2_loss(p.BFC1) + tf.nn.l2_loss(p.WFC2) + tf.nn.l2_loss(p.BFC2)

        # Loss function
        self.loss_score = tf.nn.l2_loss(self.rna_labels - self.preds, name=self.scope+'rnaloss_score')
        self.loss = self.loss_score
        #self.loss = self.loss_score + 1.0 / float(p.params['batch_size']) * p.params['beta'] * self.regularizer

        self.train_step = tf.train.AdamOptimizer(p.params['lr'], name=self.scope+'rnastep').minimize(self.loss)

        # calc accuracy
        self.accuracy = tf.pow(pearson_correlation(self.preds, self.rna_labels), 1, name=self.scope+'rnaaccuracy')


class MBPredictor:

    '''
    Init all parameters of a convolution neural network (CNN) for protein-RNA binding prediction
    '''
    def __init__(self, data_paths, params, layer, filters_set, strides_x_y, network_name, loggername):

        # Store data paths
        self.train_seq_path = data_paths[0]
        self.train_struct_path = data_paths[1]
        self.test_seq_path = data_paths[2]
        self.test_struct_path = data_paths[3]

        # Store network parameters 
        self.params = params

        # Store layer data
        self.layer = layer

        # Init hyperparameters data
        self.filters_set = filters_set       
        self.opt_filters_index = -1
        self.opt_epoch = -1
        self.weights = None
        self.biases = None
        self.kpool_x_y = None

        # Set the same strides for all filters
        self.single_strides_x_y = strides_x_y
        self.strides_x_y = None

        # Set fully connecetd layer data
        self.fc_size = -1
        self.WFC1 = None
        self.BFC1 = None
        self.WFC2 = None
        self.BFC2 = None

        # Store loggin data
        self.loggername = loggername
        self.network_name = network_name


    '''
    Init all hyperparamers 
    '''
    def init_hyperparams(self, hyper_index):
        # Compute initial weights, biases and k-pools according to filters data
        self.weights = list()
        self.biases = list()
        self.kpool_x_y = list()
        input_channels = self.layer['input_channels']
        # Get the current set of hyperparameters
        filters = self.filters_set[hyper_index]
        for filter_index in range(len(filters)):
            output_channels = int(self.layer['output_channels'] / len(filters))
            curr_filter = filters[filter_index]
            curr_x = curr_filter['size_x']
            curr_y = curr_filter['size_y']
            curr_weights = tf.Variable(tf.truncated_normal([curr_x, curr_y, input_channels, output_channels], stddev=0.1))
            curr_biases = tf.Variable(tf.zeros(output_channels))
            curr_kpool_x_y = (self.params['max_seq_len'] - curr_x + 1, 1)
            self.weights.append(curr_weights)
            self.biases.append(curr_biases)
            self.kpool_x_y.append(curr_kpool_x_y)

        # Set the same strides for all filters
        self.strides_x_y = [self.single_strides_x_y] * len(filters)

        # Set fully connecetd layer data
        #self.fc_size = len(filters) * self.layer['output_channels']
        self.fc_size = self.layer['output_channels'] 
        self.WFC1 = tf.Variable(tf.truncated_normal([self.fc_size, self.params['hidden_layer_size']], stddev=0.1))
        self.BFC1 = tf.Variable(tf.zeros(1))
        self.WFC2 = tf.Variable(tf.truncated_normal([self.params['hidden_layer_size'], 1], stddev=0.1))
        self.BFC2 = tf.Variable(tf.zeros(1))


    '''
    Evaluate performance of current model on a given dataset (validation/testing)
    '''
    def evaluate_performance(self, eval_data, eval_lengths, eval_labels, scope):

        # Create a new session 
        tf.reset_default_graph()
        new_graph = tf.Graph()
        with tf.Session(graph=new_graph) as eval_sess:

            # Import the trained network
            network = tf.train.import_meta_graph(self.network_name + '.meta')

            # Load the trained parameters
            network.restore(eval_sess, self.network_name)

            # Construct feed data using placeholder names
            rna_data = new_graph.get_tensor_by_name(scope+"rnadata:0")
            rna_lengths = new_graph.get_tensor_by_name(scope+"rnalengths:0")
            rna_labels = new_graph.get_tensor_by_name(scope+"rnalabels:0")
            feed_dict={rna_data: eval_data, rna_lengths: eval_lengths, rna_labels: eval_labels}

            # Access the evaluation metric
            eval_accuracy = new_graph.get_tensor_by_name(scope+"rnaaccuracy:0")
            result = eval_sess.run(eval_accuracy, feed_dict)

            return result

    '''
    Train a convolution neural network (CNN) for protein-RNA binding prediction
    '''
    def train(self):

        scope = "train/"
        tf.reset_default_graph()

        # Get numpy arrays with data (training and testing)
        all_train_data, all_train_lengths, all_train_labels, all_train_size = read_combined_data(self.train_seq_path,    \
                                                                                                 self.train_struct_path, \
                                                                                                 self.params['max_seq_len'])

        # Compute validation set size
        validation_size = int(all_train_size *  self.params['validation_fold'])

        # Go over all configurations of hyperparameters
        configuration_accuracies = list()
        configuration_epochs = list()
        for hyper_index in range(len(self.filters_set)):

            logger = open(self.loggername, "a")
            logger.write("Checking configuration " + str(self.filters_set[hyper_index]) + "\n")
            logger.close()

            cg = CGraph(self, hyper_index, scope)
            cg.build()

            # TODO: (1) regularization (2) dropout (3) Decay in learning rate

            # Create a new session with the configuration graph
            with tf.Session() as sess:

                # Perform k-fold cross validation early stopping and configuration evaluation
                fold_accuracies = list()
                fold_epochs = list()

                for fold_index in range(self.params['k-fold']):

                    logger = open(self.loggername, "a")
                    logger.write("\tCross Validation round " + str(fold_index + 1) + "\n")
                    logger.close()

                    # Split to traninig and validation data
                    val_start_index = fold_index * validation_size
                    val_end_index = (fold_index + 1) * validation_size

                    # Construct the two parts of the training set
                    train_data_1 = all_train_data[:val_start_index, :, :, :]
                    train_lengths_1 = all_train_lengths[:val_start_index]
                    train_labels_1 = all_train_labels[:val_start_index]

                    train_data_2 = all_train_data[val_end_index:, :, :, :]
                    train_lengths_2 = all_train_lengths[val_end_index:]
                    train_labels_2 = all_train_labels[val_end_index:]

                    # Construct the validation set
                    train_data = np.concatenate((train_data_1, train_data_2), axis=0)
                    train_lengths = np.concatenate((train_lengths_1 , train_lengths_2), axis=0)
                    train_labels = np.concatenate((train_labels_1, train_labels_2), axis=0)

                    # Construct the validation set
                    validation_data = all_train_data[val_start_index:val_end_index, :, :, :]
                    validation_lengths = all_train_lengths[val_start_index:val_end_index]
                    validation_labels = all_train_labels[val_start_index:val_end_index]

                    # Create dataset object for batching
                    dataset = Dataset(train_data, train_lengths, train_labels, self.params['batch_size'])

                    # Start trainig by initializing the session
                    init = tf.global_variables_initializer()
                    sess.run(init)

                    # Init the early-stop variables
                    max_validation_accuracy = -2.0
                    non_inc_validation_seq_len = 0
                    is_overfitting = False

                    # Perform epochs till convergence (detected by overfitting)
                    for epoch_index in range(self.params['num_epochs']):
                        batch_counter = 0
                        while dataset.has_next_batch():
                            batch_counter += 1

                            # Read next batch
                            rna_batch, lengths_batch, labels_batch = dataset.next_batch()
                            feed_dict = {cg.rna_data: rna_batch, cg.rna_lengths: lengths_batch, cg.rna_labels: labels_batch}

                            # Perform batch optimization
                            sess.run(cg.train_step, feed_dict=feed_dict)

                            # check if performance on validaiton data is decreasing enough to stop training process
                            if batch_counter % self.params['stop_check_interval'] == 0:

                                # Compute batch loss and accuracy
                                train_loss, train_accuracy = sess.run([cg.loss, cg.accuracy], feed_dict=feed_dict)

                                # Compute validation accuracy
                                validation_feed_dict = {cg.rna_data: validation_data, cg.rna_lengths: validation_lengths, cg.rna_labels: validation_labels}
                                validation_accuracy = cg.accuracy.eval(feed_dict=validation_feed_dict)

                                # statistics
                                status = 'Train acc: {}\tTrain Loss: {}\tValidation acc: {}\t{}\t{}\t{}'\
                                                     .format(train_accuracy, train_loss, validation_accuracy, \
                                                             batch_counter, dataset.curr_ind, epoch_index)
                                print (status)

                                logger = open(self.loggername, "a")
                                logger.write("\t\t" + str(validation_accuracy) + "\t" + str(dataset.curr_ind) + "\t" + str(epoch_index) + "\n")
                                logger.close()
 
                                # Upadte optimal validation accuracy (if needed)
                                if validation_accuracy > max_validation_accuracy:
                                    max_validation_accuracy = validation_accuracy
                                    non_inc_validation_seq_len = 0
                                else:
                                    non_inc_validation_seq_len += 1

                                # Early stopping in case of continuous decrease in validation accuracy
                                if non_inc_validation_seq_len == self.params['early_stopping_memory']:
                                    is_overfitting = True
                                    break

                        # Reset dataset for a possible next epoch            
                        dataset.reset()

                        # Terminate epochs if overfitting on validation set has been detected
                        if is_overfitting: 
                            break
 
                    # Compute the final validation accuracy for this cross validation (training and validation pair)
                    cv_acc = cg.accuracy.eval(feed_dict={cg.rna_data: validation_data, cg.rna_lengths: validation_lengths, cg.rna_labels: validation_labels})
                    fold_accuracies.append(cv_acc)
                    fold_epochs.append(epoch_index + 1)
                    print ('\t\tFinal validation_acc=', cv_acc)
  
                # Compute average validation accuracy for a specific hyperparameters configuration
                config_accuracy = np.array(fold_accuracies).mean()
                config_epochs = int(np.ceil(np.array(fold_epochs).mean()))
                print ("Mean Validation accuracy for configuration=", config_accuracy)
                configuration_accuracies.append(config_accuracy)
                configuration_epochs.append(config_epochs)
                logger = open(self.loggername, "a")
                logger.write("Mean configuration accuracy " + str(config_accuracy) + "\n\n")
                logger.write("Mean configuration epochs " + str(config_epochs) + "\n\n")
                logger.close()

        # Get the optimal choice for the hyperparameters
        self.opt_filters_index = np.argmax(configuration_accuracies)
        self.opt_epochs = configuration_epochs[self.opt_filters_index]

        print (self.opt_filters_index, self.opt_epochs)
        logger = open(self.loggername, "a")
        logger.write("Training optimum hyperparameters are " + str(self.opt_filters_index) + "\t" + str(self.opt_epochs) + "\n\n\n")
        logger.close()



    def final_train(self):

        scope = "final-train/"
        tf.reset_default_graph()

        # Get numpy arrays with data (training and testing)
        all_train_data, all_train_lengths, all_train_labels, all_train_size = read_combined_data(self.train_seq_path,    \
                                                                                                 self.train_struct_path, \
                                                                                                 self.params['max_seq_len'])
        logger = open(self.loggername, "a")
        logger.write("Train on all data\n\n")
        logger.close()

        # Run CNN network
        cg = CGraph(self, self.opt_filters_index, scope)
        cg.build()

        # Create a saver object to store optimal traininig network
        saver = tf.train.Saver()

        # TODO: (1) regularization (2) dropout (3) Decay in learning rate

        with tf.Session() as sess:

            # Create dataset object for batching
            dataset = Dataset(all_train_data, all_train_lengths, all_train_labels, self.params['batch_size'])

            # Start trainig by initializing the session
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch_index in range(self.opt_epochs):
                batch_counter = 0
                while dataset.has_next_batch():
                    batch_counter += 1

                    # Read next batch
                    rna_batch, lengths_batch, labels_batch = dataset.next_batch()
                    feed_dict = {cg.rna_data: rna_batch, cg.rna_lengths: lengths_batch, cg.rna_labels: labels_batch}

                    # Perform batch optimization
                    sess.run(cg.train_step, feed_dict=feed_dict)

                    # check if performance on trianing data is decreasing enough to stop training process
                    if batch_counter % self.params['stop_check_interval'] == 0:

                        # Compute batch loss and accuracy
                        train_loss, train_accuracy = sess.run([cg.loss, cg.accuracy], feed_dict=feed_dict)

                        # statistics
                        status = 'Train acc: {}\tTrain Loss: {}\t{}\t{}\t{}'\
                                                 .format(train_accuracy, train_loss,\
                                                         batch_counter, dataset.curr_ind, epoch_index)
                        print (status)

                        logger = open(self.loggername, "a")
                        logger.write(str(train_accuracy) + "\t" + str(dataset.curr_ind) + "\t" + str(epoch_index) + "\n")
                        logger.close()

                dataset.reset()

            save_path = saver.save(sess, self.network_name)


    '''
    Test trained network for protein-RNA binding prediction
    '''
    def test(self):

        test_data, test_lengths, test_labels, test_size = read_combined_data(self.test_seq_path, \
                                                                             self.test_struct_path, \
                                                                             self.params['max_seq_len'])

        result = self.evaluate_performance(test_data, test_lengths, test_labels, "final-train/")

        # Remove network files
        #os.remove(self.network_name + ".data-00000-of-00001")
        #os.remove(self.network_name + ".index")
        #os.remove(self.network_name + ".meta")

        return result

