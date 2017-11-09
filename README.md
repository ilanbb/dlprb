# dlprb
DLPRB: A Deep Learning Approach for Predicting Protein-RNA Binding

Implementation of two deep neural networks (DNN's): a convolutional neural netwrk (CNN) and a recurrent neural network (RNN).

Requirements:
Python, Numpy, Tensorflow.

Setting up:
1. Clone the repositopry into your working space.
2. for both cnn and rnn directories:
	2a. Change the DATA_DIR variable in main.py to point to the actual input files directory.
	2b. Change the following variables in main.py according to the actual input file name patterns: all_files, TRAIN_STRUCTURE_FILE, TEST_STRUCTURE_FILE, TRAIN_SEQUENCE_FILE, TERST_SEQUENCE_FILE.

Training:
For traininig a CNN architecture: python3.4 cnn/main.py train
For traininig an RNN architecture: python3.4 rnn/main.py train

Testing:
For testing pre-trained CNN architecture: python3.4 cnn/main.py test
For testing pre-trained RNN architecture: python3.4 rnn/main.py test

Input format:
A sequence-information file contains a set of sequnces together with their estimated binding affinities. 
Every line in the file is of the following form: <binding-affinity> <RNA-sequence>

A structure-information file contains an RNA secondary structure annotations of the sequences estimated. The structure information of a seuqence is encoded within a block of six lines. 
The first line stores the sequence itself, preceded with a '>' sign.
The next five lines of the block encodes the structural contexts probabilities at each position, every row corresponds to a different structural context: paired, hairpin loop, internal loop, multiloop, and external loop.

The data directory should contain all experiments files. Every experiment has four related files: 
- Train sequence-information file.
- Train structure-information file.
- Test sequence-information file.
- Test structure-information file.

