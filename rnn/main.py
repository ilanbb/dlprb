from model import *
import time
import glob
import sys
import os

# Random generator initializers
tf.set_random_seed(1)
np.random.seed(1)

if len(sys.argv) != 3:
    print ("Usage: python3.4 main.py <train/test> <data_dir>")
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

# Set strides
strides_x_y = (1, 1)

network_name = './network'

# Set data paths
directive = sys.argv[1]
data_dir = sys.argv[2]

# Add trailing dir separator
if not data_dir.endswith(os.path.sep):
    data_dir = data_dir + os.path.sep

# Set loggin options
loggername = "results-" + __file__[:__file__.index('.')] + ".txt"

struct_files = glob.glob(data_dir + '*annotations*')
sequence_files = glob.glob(data_dir + '*sequences*')

if len(struct_files) == 0 or len(sequence_files) == 0:
    print ("Warning: no input files found!")
    exit()

for seq_file, struct_file in zip(sequence_files, struct_files):

    print (seq_file + "\n" + struct_file)
    start_time = time.time()

    data_paths = (seq_file, struct_file)

    # Initialize a RNN network for motif binding prediction
    net_file = network_name + "-" + seq_file[len(data_dir):]
    predictor = MBPredictor(data_paths, params, net_file, loggername)

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
