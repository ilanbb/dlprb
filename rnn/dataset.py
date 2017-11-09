import numpy as np
from sklearn.utils import shuffle


class Dataset():
    """
    Some naive implementation to get training batches
    """

    def __init__(self, rna_sequences, lengths, labels, batch_size):
        self.rna_sequences = rna_sequences
        self.lengths = lengths
        self.labels = labels
        self.batch_size = batch_size
        self.curr_ind = 0
        self.batch_size = batch_size
        self.data_size = rna_sequences.shape[0]

    def has_next_batch(self):
        return self.curr_ind != self.data_size

    def reset(self):
        self.curr_ind = 0
        # self.x1, self.x2, self.y = shuffle(self.x1, self.x2, self.y, random_state=0)  # shuffle training data

    def next_batch(self):
        end_ind = self.curr_ind + self.batch_size

        if end_ind > self.data_size:
            end_ind = self.data_size

        rnas_batch = self.rna_sequences[self.curr_ind:end_ind, :]
        lengths_batch = self.lengths[self.curr_ind:end_ind]
        labels_batch = self.labels[self.curr_ind:end_ind]

        self.curr_ind = end_ind

        return rnas_batch, lengths_batch, labels_batch
