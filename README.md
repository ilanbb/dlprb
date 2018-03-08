# dlprb
DLPRB: A Deep Learning Approach for Predicting Protein-RNA Binding

A TensorFlow implementation of two deep neural networks (DNN's): a convolutional neural netwrk (CNN) and a recurrent neural network (RNN).
The main.py files are currently adjusted for analyzing the RNAcompete dataset of 244 experiments.

Requirements:

	Python, Numpy, Tensorflow.

Setting up:

	Clone the repositopry into your working space.

Training:

	For traininig a CNN architecture (from the cnn directory): python main.py train <train-data-dir>

	For traininig an RNN architecture (from the rnn directory): python main.py train <train-data-dir>

Testing:

	For testing pre-trained CNN architecture (from the cnn directory): python main.py test <test-data-dir>

	For testing pre-trained RNN architecture (from the rnn directory): python main.py test <test-data-dir>

File names:

	Sequence-information file should have the string "sequences" in its name.
	Structure-information file should have the string "annotations" in its name.
	Training and testing files of the same experiment should have the same name (in two different directories).

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

Every experiment has four possible related files (two for training and two for testing).

The train data directory should contain the following files:
- Train sequence-information file.
- Train structure-information file.

The test data directory should contain the following files:
- Test sequence-information file.
- Test structure-information file.

When testing, the directory including the main.py file should also contain the files produced by the training module.

