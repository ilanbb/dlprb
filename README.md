# dlprb
DLPRB: A Deep Learning Approach for Predicting Protein-RNA Binding

A TensorFlow implementation of two deep neural networks (DNN's): a convolutional neural netwrk (CNN) and a recurrent neural network (RNN).

Requirements:

	Python, Numpy, Tensorflow.

Setting up:

	1. Clone the repositopry into your working space.

	2. for both cnn and rnn directories:

		2a. Change the DATA_DIR variable in main.py to point to the actual input files directory.

		2b. Change the following variables in main.py according to the actual input file name patterns: 
			(*) all_files
			(*) TRAIN_STRUCTURE_FILE
			(*) TEST_STRUCTURE_FILE
			(*) TRAIN_SEQUENCE_FILE
			(*) TERST_SEQUENCE_FILE

Training:

	For traininig a CNN architecture: python main.py train (from the cnn directory)

	For traininig an RNN architecture: python main.py train (from the rnn directory)

Testing:

	For testing pre-trained CNN architecture: python main.py test (from the cnn directory)

	For testing pre-trained RNN architecture: python main.py test (from the rnn directory)

Input format:

A sequence-information file contains a set of sequnces together with their estimated binding affinities. Every line in the files starts with binding affinity and then followed by the RNA sequence. For example:

	-0.975763 AGAAGGCACCAACAGAAGCUCUAACCAGACUAGCCACC

A structure-information file contains an RNA secondary structure annotations of the sequences estimated. The structure information of a seuqence is encoded within a block of six lines. 
The first line stores the sequence itself, preceded with a '>' sign.
The next five lines of the block encodes the structural contexts probabilities at each position, every row corresponds to a different structural context: paired, hairpin loop, internal loop, multiloop, and external loop. For example:

	>AGAACU
	0.0023	0.0208	0.0031	0.0041	0.987	0.9878
	0.0000	0.0000	0.0004	0.0004	0.0004	0.0044
	0.0000	0.0000	0.0179	0.0207	0.0027	0.0003
	0.0000	0.0000	0.0010	0.0010	0.0003	0.0001
	0.9977	0.9792	0.9776	0.9738	0.0096	0.0074

The data directory should contain all experiments files. Every experiment has four related files: 
- Train sequence-information file.
- Train structure-information file.
- Test sequence-information file.
- Test structure-information file.

