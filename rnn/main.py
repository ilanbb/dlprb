from model import *
import time
import glob
import sys

# Random generator initializers
tf.set_random_seed(1)
np.random.seed(1)

if len(sys.argv) != 2:
    print ("Usage: python3.4 main.py <train/test>")
    exit()

# Set loggin options
loggername = "results-" + __file__[:__file__.index('.')] + ".txt"

# Set model parameters
params = dict()
params['max_seq_len'] = 41
params['structures'] = 5
params['batch_size'] = 128
params['beta'] = 0.001
params['lr'] = 0.0002
params['num_epochs'] = 18
params['hidden_layer_size'] = 128
params['stop_check_interval'] = 50
params['rnn_cell_size'] = 64

# Set the layers data
layer1 = {'input_channels': 2, 'output_channels': 512}
#layer1 = {'input_channels': 2, 'output_channels': 252}

# Set the filters data
filter11 = {'size_x':5, 'size_y': params['structures']}
filter12 = {'size_x':6, 'size_y': params['structures']}
filter13 = {'size_x':7, 'size_y': params['structures']}
filter14 = {'size_x':8, 'size_y': params['structures']}
filter15 = {'size_x':9, 'size_y': params['structures']}
filter16 = {'size_x':4, 'size_y': params['structures']}
filter17 = {'size_x':10, 'size_y': params['structures']}
filter18 = {'size_x':11, 'size_y': params['structures']}
filter19 = {'size_x':16, 'size_y': params['structures']}
filters_1 = [filter11, filter12, filter13, filter14, filter15, filter16, filter17, filter18]
filters_2 = [filter11, filter14, filter18]
filters_3 = [filter11]
filters_4 = [filter17]
filters_5 = [filter19]
filters_6 = [filter11, filter17, filter19]
filters_7 = [filter18]
filters_8 = [filter11, filter18]
filters_set = [filters_3, filters_4, filters_5, filters_6]
filters_set = [filters_3, filters_7, filters_8]

# Set strides
strides_x_y = (1, 1)

network_name = './network'

# Set data paths
DATA_DIR = "../data/"

# Set loggin options
loggername = "results-" + __file__[:__file__.index('.')] + ".txt"

all_files = glob.glob(DATA_DIR + 'RNCMPT00[0-9][0-9][0-9].txt.annotations_A.RNAcontext-sample')

if len(all_files) == 0:
    print ("Warning: no input files found!")
    exit()

exp_indices = list()
for file_name in all_files:
    counter_start_index = len(DATA_DIR) + 8 
    exp_index = file_name[counter_start_index : counter_start_index + 3]
    if exp_index[0] == '0' and exp_index[1] == '0':
        idx = int(exp_index[2])
    elif exp_index[0] == '0':
        idx = int(exp_index[1:])
    else:
        idx= int(exp_index)
    exp_indices.append(exp_index)

exp_indices.sort()

directive = sys.argv[1]

for exp_index in exp_indices:
    print (exp_index)
    start_time = time.time()

    str_exp_index = str(exp_index)
    diff = 3 - len(str_exp_index)
    prefix = diff * '0'
    str_exp_index = prefix + str_exp_index

    TRAIN_STRUCTURE_FILE= "RNCMPT00" + str_exp_index + ".txt.annotations_A.RNAcontext-sample"
    TEST_STRUCTURE_FILE = "RNCMPT00" + str_exp_index + ".txt.annotations_B.RNAcontext-sample"
    TRAIN_SEQUENCE_FILE = "RNCMPT00" + str_exp_index + ".txt.sequences_A.RNAcontext.clamp-sample"
    TEST_SEQUENCE_FILE = "RNCMPT00" + str_exp_index + ".txt.sequences_B.RNAcontext.clamp-sample"

    train_seq = DATA_DIR + TRAIN_SEQUENCE_FILE
    train_struct = DATA_DIR + TRAIN_STRUCTURE_FILE
    test_seq = DATA_DIR + TEST_SEQUENCE_FILE
    test_struct = DATA_DIR + TEST_STRUCTURE_FILE
    data_paths = (train_seq, train_struct, test_seq, test_struct)

    # Initialize a RNN network for motif binding prediction
    predictor = MBPredictor(data_paths, params, network_name + "-" + str_exp_index, loggername)

    if directive == "train":
        # Hyperparameter optimization
        predictor.train()

    elif directive == "test":
        # Test the model
        result = predictor.test()
        print (result)

    else:
         print ("Unknown directive")
         break

    end_time = time.time()
    duration = end_time - start_time
    print (duration)
