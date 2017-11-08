import numpy as np
import tensorflow as tf


'''
Helper function to construct a convolutional layer given its parameres
'''
def create_cnn_layer(data, weights_matrix, bias_vector, strides_x_y):
    all_strides = [1, strides_x_y[0], strides_x_y[1], 1]
    result = tf.nn.conv2d(data, weights_matrix, strides=all_strides, padding='VALID')
    result = tf.nn.bias_add(result, bias_vector)
    result = tf.nn.relu(result)
    return result

'''
Helper function to construct a pooling layer given its parameres
'''
def create_pooling_layer(data, kpool_x_y, strides_x_y):
    all_kpools = [1, kpool_x_y[0], kpool_x_y[1], 1]
    all_strides = [1, strides_x_y[0], strides_x_y[1], 1]
    result = tf.nn.max_pool(data, ksize=all_kpools, strides=all_strides, padding='VALID')
    return result

'''
Helper function to construct two layers of convolution and max pooling
'''
def create_layer(data, weights_matrix, bias_vector, strides_x_y, kpool_x_y):
    result = create_cnn_layer(data, weights_matrix, bias_vector, strides_x_y)
    result = create_pooling_layer(result, kpool_x_y, strides_x_y)
    return result

'''
Helper function to construct a CNN (convolution and pooling) with multiple filters
'''
def create_multiple_filter_cnn(data, layer, filters, weights, biases, strides_x_y, kpool_x_y):
    pooled_outputs = []
    for filter_index in range(len(weights)):
        f = filters[filter_index]
        W = weights[filter_index]
        b = biases[filter_index]
        s = strides_x_y[filter_index]
        k = kpool_x_y[filter_index]
        filter_output = create_layer(data, W, b, s, k)
        # Second dimension of out depends on the the filter size, so padding might be needed
        #diff = ... params['max_seq_len'] - filters[filter_index]['size_x'] + 1
        #if diff > 0:
        #    pad = tf.zeros([tf.shape(filter_output)[0], f['size_x'] - 1, 1, layer['output_channels']], tf.float32)
        #    filter_output = tf.concat([filter_output, pad], axis=1)
        print ("FILTER OUTPUT", tf.shape(filter_output), filter_output.get_shape())
        #tf.concat([filter_output, zeros], axis=0)
        pooled_outputs.append(filter_output)
    # Final output of lavel is of size [BATCH_SIZE, MAX_SEQ_LEN / POOL_X (1), STRUCTURES / POOL_Y (1), FILTERS * OUPUT_CHANNELS]
    cnn_output = tf.concat(pooled_outputs, 3)
    #pool = tf.squeeze(pool)
    #print ("POOL", tf.shape(pool), pool.get_shape())
    return cnn_output


'''
Helper function to flatten a CNN output
'''
def flatten(conv_data, fc_size):
    flat_data = tf.reshape(conv_data, [-1, fc_size])
    return flat_data

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

