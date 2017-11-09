import numpy as np

STRUCTURES = 5

# Process next CLAMP sequence data: get sequence and binding affinity
# A CLAMP line looks like: SCORE SEQUQNCE
def process_clamp(clamp_file):   
    data = clamp_file.readline()
    if not data:
        return None
    line = data.strip().split()
    score = float(line[0])
    seq = line[1]
    return (seq, score)

# Process next RNAcontext sequence data: get sequnce and (STRUCTURES X SEQ_LEN) matrix
def process_rnacontext(rnacontext_file):
    data = rnacontext_file.readline()
    if not data:
        return None
    seq_line = data.strip()
    assert (seq_line[0] == '>')
    seq = seq_line[1:]
    matrix = list()
    for structure_index in range(STRUCTURES):
        structure_line = rnacontext_file.readline().strip()
        matrix_line = [float(elem) for elem in structure_line.split()]
        matrix.append(matrix_line)
    return (seq, matrix)

def read_sequence_only(sequences_path, structures_path, max_seq_len):
    with open(sequences_path, 'r') as sequences:
        data = list()
        lengths = list()
        labels = list()
        counter = 0
        while True:
            counter += 1
            seq_data = process_clamp(sequences)
            if not seq_data:
                return np.array(data), np.array(lengths), np.array(labels)
            # Compute a matrix of SEQ_LEN X RNA_ALPHABET for decoding the sequence bases
            labels.append(seq_data[1])
            seq_matrix = list()
            for base in seq_data[0]:
                if base == 'A':
                    base_encoding = [1, 0, 0, 0]
                elif base == 'C':
                    base_encoding = [0, 1, 0, 0]
                elif base == 'G':
                    base_encoding = [0, 0, 1, 0]
                elif base == 'U':
                    base_encoding = [0, 0, 0, 1]
                else:
                    raise ValueError
                seq_matrix.append(base_encoding)
            seq_matrix = np.array(seq_matrix)
            # Vertical padding: equal the number of columns in both channels
            ver_diff = STRUCTURES - seq_matrix.shape[1]
            assert (ver_diff >= 0)
            if ver_diff > 0:
                padding_columns = np.zeros((seq_matrix.shape[0], ver_diff))
                seq_matrix = np.concatenate((seq_matrix, padding_columns), axis=1)
            # Horizontal Padding: each RNA seq should be of MAX_SEQ_LEN
            curr_seq_len = seq_matrix.shape[0]
            lengths.append(curr_seq_len)
            padd_len = max_seq_len - curr_seq_len
            assert (padd_len  >= 0)
            if padd_len > 0:
                padding_matrix = np.zeros((padd_len, STRUCTURES))
                seq_matrix = np.concatenate((seq_matrix, padding_matrix), axis=0)
            base_matrix = np.dstack((seq_matrix, seq_matrix))
            #print (base_matrix.shape)
            data.append(base_matrix)


def read_combined_data(sequences_path, structures_path, max_seq_len):
    with open(sequences_path, 'r') as sequences, open(structures_path, 'r') as structures:
        data = list()
        lengths = list()
        labels = list()
        counter = 0
        while True:
            counter += 1
            seq_data = process_clamp(sequences)
            structure_data = process_rnacontext(structures)
            if not seq_data or not structure_data:
                return np.array(data), np.array(lengths), np.array(labels), counter-1
            #print ("Line", counter)
            #print (seq_data)
            #print (structure_data)
            # Compute a matrix of SEQ_LEN X RNA_ALPHABET for decoding the sequence bases
            labels.append(seq_data[1])
            seq_matrix = list()
            for base in structure_data[0]:
                if base == 'A':
                    base_encoding = [1, 0, 0, 0]
                elif base == 'C':
                    base_encoding = [0, 1, 0, 0]
                elif base == 'G':
                    base_encoding = [0, 0, 1, 0]
                elif base == 'U':
                    base_encoding = [0, 0, 0, 1]
                else:
                    raise ValueError
                seq_matrix.append(base_encoding)
            seq_matrix = np.array(seq_matrix)
            # Compute a matrix of SEQ_LEN X STRUCTURE for decoding the sequence structures 
            struct_matrix = np.transpose(np.array(structure_data[1]))
            #print ("==")
            #print (base_matrix.shape)
            # Vertical padding: equal the number of columns in both channels
            ver_diff = STRUCTURES - seq_matrix.shape[1]
            assert (ver_diff >= 0)
            if ver_diff > 0:
                padding_columns = np.zeros((seq_matrix.shape[0], ver_diff))
                seq_matrix = np.concatenate((seq_matrix, padding_columns), axis=1)
            # Horizontal Padding: each RNA seq should be of MAX_SEQ_LEN
            curr_seq_len = seq_matrix.shape[0]
            lengths.append(curr_seq_len)
            padd_len = max_seq_len - curr_seq_len
            assert (padd_len  >= 0)
            if padd_len > 0:
                padding_matrix = np.zeros((padd_len, STRUCTURES))
                seq_matrix = np.concatenate((seq_matrix, padding_matrix), axis=0)
                struct_matrix = np.concatenate((struct_matrix, padding_matrix), axis=0)
            base_matrix = np.dstack((seq_matrix, struct_matrix))
            #print (base_matrix.shape)
            data.append(base_matrix)            
        assert(False)
    

# Testing code
if __name__ == "__main__":
    
    DATA_DIR = "/specific/a/home/cc/students/cs/shiranabadi/motif-binding/"

    TRAIN_STRUCTURE_FILE= "RNCMPT00001.txt.annotations_A.RNAcontext"
    TEST_STRUCTURE_FILE	= "RNCMPT00001.txt.annotations_B.RNAcontext"
    TRAIN_SEQUENCE_FILE = "RNCMPT00001.txt.sequences_A.RNAcontext.clamp" 
    TEST_SEQUENCE_FILE = "RNCMPT00001.txt.sequences_B.RNAcontext.clamp" 

    MAX_SEQ_LEN = 41

    data, lengths, labels = read_combined_data(DATA_DIR + TRAIN_SEQUENCE_FILE, DATA_DIR + TRAIN_STRUCTURE_FILE, MAX_SEQ_LEN)
    print (data.shape)
    print (lengths.shape)
    print (labels.shape)
    print (data[0])
    print (lengths[0])
    print (labels[0])

