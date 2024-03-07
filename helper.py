import numpy as np
from scipy.sparse import csr_matrix
import re
from plot_cluster import *
from Bio.Align.Applications import ClustalOmegaCommandline
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import sklearn
import torch
from sklearn.metrics import pairwise_distances
from collections import Counter

############################## CLUSTAL ##############################
def runClustalRange(architecture_name, start, end, unique_labels, verbose=False ):
    # Requires clustal to be installed
    # runs clustalo -i cluster_0.fasta -o output_0.clustal -outfmt=clustal for each cluster
    # clustal files needed for plotClusters and evaluate gaps
    for ind in range(start, end):
        input_fasta = f"clusters/{architecture_name}/cluster_{unique_labels[ind]}.fasta"
        clustal_file = f"clustal/{architecture_name}/cluster_{unique_labels[ind]}.clustal"
        clustalomega_cline = ClustalOmegaCommandline(infile=input_fasta, outfile=clustal_file, outfmt='clustal', force=True, verbose=True)

        stdout, stderr = clustalomega_cline()

        if verbose:
            print(stdout)

def runClustal(architecture_name, unique_labels, verbose=False):
    # Requires clustal to be installed
    # runs clustalo -i cluster_0.fasta -o output_0.clustal -outfmt=clustal for each cluster
    # clustal files needed for plotClusters and evaluate gaps
    runClustalRange(architecture_name, 0, len(unique_labels), unique_labels, verbose)

def evaluateGapsInRange(architecture_name, start, end, unique_labels):
    gap_percentages = []
    for ind in range(start, end):
        clustal_file = f"clustal/{architecture_name}/cluster_{unique_labels[ind]}.clustal"
        print(clustal_file)

        with open(clustal_file, 'r') as file:
            lines = file.readlines()

            # strip Seq_X and take the sequence with line.split()[1]
            alignments = []
            for line in lines:
                if len(line.split()) == 2 and not line.startswith('CLUSTAL'):
                    alignments.append(line.split()[1])
            # [line.split()[1] for line in lines if len(line.strip()) != 0 and not line.startswith('CLUSTAL')

            alignment_length = len(alignments[0])

            # count number of "-" in the clustal file
            gap_count = sum(seq.count('-') for seq in alignments)
            total_positions = len(alignments) * alignment_length

            gap_percentage = (gap_count / total_positions) * 100
            gap_percentages.append(gap_percentage)

        print(f"Percentage of gaps in the alignment for cluster {unique_labels[ind]}: {gap_percentage:.2f}%")
    return np.array(gap_percentages)

def evaluateGaps(architecture_name, unique_labels):
    return evaluateGapsInRange(architecture_name, 0, len(unique_labels), unique_labels)

def countPercentageLowerThan(percentages, x):
    return np.count_nonzero(percentages < x)

def getFamiliesPerCluster(cluster_labels, unique_labels, id_list, ID_family_mapping):
    # get ids of sequences from each cluster
    ids_per_cluster = {}
    for cluster_label in unique_labels:
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_label]
        cluster_ids = [id_list[i] for i in cluster_indices]
        ids_per_cluster[cluster_label] = cluster_ids

    # get families
    family_frequencies_per_cluster = {}
    family_count_per_cluster = {}
    for cluster_label, cluster_ids in ids_per_cluster.items():
        family_frequencies = {}

        for id in cluster_ids:
            family = ID_family_mapping.get(id)
            if family:
                family_frequencies[family] = family_frequencies.get(family, 0) + 1

        family_frequencies_per_cluster[cluster_label] = family_frequencies
        family_count_per_cluster[cluster_label] = len(family_frequencies.keys())
        
    return family_count_per_cluster, family_frequencies_per_cluster


def getIDFamilyMapping(seq_records):
    id_to_family = {}
    for record in seq_records:
        record_id = record.id
        description = record.description
        family_name = description.split()[2]
        id_to_family[record_id] = family_name

    return id_to_family

def primerExtractionScore(clustal_file, window_size = 50, stride = 5, target_consensus = 2/3*100, max_gap_percentage = 25):
    # take strings of sequences from clustal file
    sequences = list(SeqIO.parse(clustal_file, "clustal"))
    alignments = np.array([list(str(seq.seq)) for seq in sequences]).T

    # go through each window in the clustal file
    consensus_percentages = []
    gap_percentages = []
    for i in range(0, alignments.shape[0] - window_size, stride):
        window = alignments[i:i + window_size, :]

        # calculate gap percentage for each sequence in the window
        gap_percentage = np.sum(window == '-')/(alignments.shape[1]*window_size) * 100

        if gap_percentage > max_gap_percentage:
            continue

        gap_percentages.append(gap_percentage)

        # sum occurrences of non-gap characters for all sequences
        consensus_count = 0
        for j in range(window.shape[0]):
            seq_without_gaps = [c for c in window[j] if c != '-']
            if seq_without_gaps:
                counts = Counter(seq_without_gaps)
                consensus_count += counts.most_common(1)[0][1]

        # consensus_count / window_size is percentage per window
        # divide by window.shape[0] to get percentage on all sequences
        consensus_percentage = (consensus_count / window_size) * 100 / window.shape[1]
        if consensus_percentage >= target_consensus:
            consensus_percentages.append(consensus_percentage)

    if len(consensus_percentages) == 0:
        return 0
    return np.max(consensus_percentages)

def primerExtractionScoreOnModel(architecture_name, cluster_labels, unique_labels, window_size = 50, stride = 5, target_consensus = 2/3*100, max_gap_percentage = 25):
    scores = []
    for ind in range(0, len(unique_labels)):
        clustal_file = f"clustal/{architecture_name}/cluster_{unique_labels[ind]}.clustal"
        scores.append(primerExtractionScore(clustal_file, stride = 1))

    _, label_counts = np.unique(cluster_labels, return_counts=True)
    label_percentages = label_counts / len(cluster_labels)

    normalized_scores = scores * label_percentages
    sum_scores = np.sum(normalized_scores)
    div_scores = sum_scores/len(unique_labels)

    return scores, normalized_scores, sum_scores, div_scores

def plotClusterRange(architecture_name, start, end, unique_labels):
    # Run Antoine's plotting for each cluster
    for ind in range(start, end):
        clustal_file = f"clustal/{architecture_name}/cluster_{unique_labels[ind]}.clustal"
        plot_png = f"plots/{architecture_name}/cluster_{unique_labels[ind]}.png"
        print(f"Plot for cluster {ind}")
        plot_consensus(clustal_file, plot_png)

def plotAllClusters(architecture_name, unique_labels):
    # Run Antoine's plotting for each cluster
    plotClusterRange(architecture_name, 0, len(unique_labels), unique_labels)

############################## KMeans ##############################

def KMeansOnEmbeddings(sequences_list, model, n_clusters, seed=42, normalize=True):
    # for each data point, put it into the model, get embeddings (10000, 1, 128)
    embeddings = []
    for seq in sequences_list:
        embedded_seq = model(seq.unsqueeze(0))
        embedded_seq = sklearn.preprocessing.normalize(embedded_seq.detach().numpy())
        embeddings.append(embedded_seq)

    # reshape that into (10000, 128)
    embeddings = np.concatenate(embeddings, axis=0)

    # define Kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, max_iter=1500)
    
    # run Kmeans
    cluster_labels = kmeans.fit_predict(embeddings)

    # count labels to see if the clusters look okay ish
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)

    for label, count in zip(unique_labels, label_counts):
        print(f"Label {label}: Count {count}")

    return cluster_labels, unique_labels

def pairWiseCrossValidation(data, sequences, model):
    sum_means = []
    for k in range(3, 50):
        cluster_labels, unique_labels = KMeansOnEmbeddings(data, model, k)
        means = []
        for ind in unique_labels:
            cluster_sequences = [sequences[i] for i, label in enumerate(cluster_labels) if label == unique_labels[ind]]
            
            means.append(np.mean(pairwise_distances(cluster_sequences, metric="cosine")))
        sum_means.append(np.mean(np.array(means)))
    return np.argmin(sum_means) + 3

############################## Get Data ##############################

def getData(file_path, num_sequences):
    encoded_sequences = []
    
    i = 0
    with open(file_path, 'r') as file:
        for line in file:
            if i >= num_sequences:
                return np.array(encoded_sequences)

            if "[" in line and "]" in line:
                matches = re.findall(r'\[(.*?)\]', line)
                encoded_seq = [list(map(float, match.split())) for match in matches]
                encoded_sequences.append(np.array(encoded_seq))
            else:
                encoded_seq = list(map(float, line.strip().split()))
                encoded_sequences.append(encoded_seq)
            i += 1
    
    return np.array(encoded_sequences)

def getIDs(file_path):
    ids = np.genfromtxt(file_path, dtype='str')
    return ids

############################## Write Data ##############################

def writeEncoding(output_file, encoded_sequences):
    with open(output_file, "w") as file:
        for sequence in encoded_sequences:
            sequence_str = ' '.join(map(str, sequence))
            file.write(sequence_str + '\n')

def writeIDs(sequences_list, file_path):
    with open(file_path, 'w') as file:
        for seq in sequences_list:
            file.write(seq.id + '\n')

############################## Encoders ##############################

def encodeSequences(sequences_list, encoding_function, take_n_bases, apply_padding=False, max_sequence_length=0):
    encoded_sequences = []
    shapes=[]
    for seq in sequences_list:
        encoded_seq = encoding_function(seq.seq[:take_n_bases])
        if apply_padding:
            encoded_seq = applyPadding(encoded_seq, max_sequence_length)
        encoded_sequences.append(encoded_seq)
        shapes.append(encoded_seq.shape)

    return np.stack(encoded_sequences)

def decodeSequences(architecture_name, sequences_list, decoding_function, unique_labels, cluster_labels):
    for cluster_label in unique_labels:
        cluster_sequences = [sequences_list[i] for i, label in enumerate(cluster_labels) if label == cluster_label]
        
        decoded_sequences = []
        for seq in cluster_sequences:
            decoded_seq = decoding_function(seq)
            decoded_sequences.append(decoded_seq)
        
        with open(f'clusters/{architecture_name}/cluster_{cluster_label}.fasta', 'w') as output_file:
            for idx, sequence in enumerate(decoded_sequences):
                output_file.write(f'>Seq_{idx}\n{sequence}\n')

def decodeSequencesRange(architecture_name, start, end, sequences_list, decoding_function, unique_labels, cluster_labels):
    for ind in range(start, end):
        cluster_sequences = [sequences_list[i] for i, label in enumerate(cluster_labels) if label == unique_labels[ind]]
        
        decoded_sequences = []
        for seq in cluster_sequences:
            decoded_seq = decoding_function(seq)
            decoded_sequences.append(decoded_seq)
        
        with open(f'clusters/{architecture_name}/cluster_{unique_labels[ind]}.fasta', 'w') as output_file:
            for idx, sequence in enumerate(decoded_sequences):
                output_file.write(f'>Seq_{idx}\n{sequence}\n')

def applyPadding(encoded_seq, max_length):
    # csr_matrixes
    if isinstance(encoded_seq, csr_matrix):
        padding_length = max_length - encoded_seq.shape[0]
        if padding_length > 0:
            pad_left = padding_length // 2
            pad_right = padding_length - pad_left
            encoded_seq = csr_matrix(np.pad(encoded_seq.toarray(), ((pad_left, pad_right), (0, 0)), mode='constant', constant_values=0))
        return encoded_seq
    # everything else
    else:
        padding_length = max_length - len(encoded_seq)
        if padding_length > 0:
            pad_left = padding_length // 2
            pad_right = padding_length - pad_left
            encoded_seq = np.pad(encoded_seq, (pad_left, pad_right), mode='constant', constant_values=(0.0, 0.0))
        return encoded_seq

def seqToString(sequence):
    seq_string = str(sequence).lower()
    seq_string = re.sub('[^acgt]', 'n', seq_string)
    seq_string = np.array(list(seq_string))
    return seq_string

def ordinal_encoder(sequence):
    seq_string = seqToString(sequence)
    
    mapping = {
        'a': 1.00,
        'c': 2.00,
        'g': 3.00,
        't': 4.00,
        'n': 0.00
    }
    
    float_encoded = np.array([mapping[label] for label in seq_string])
    return float_encoded

def one_hot_encoder(sequence):
    seq_string = seqToString(sequence)
    encoding = {'a': [1, 0, 0, 0, 0], 't': [0, 1, 0, 0, 0], 'c': [0, 0, 1, 0, 0], 'g': [0, 0, 0, 1, 0], 'n': [0, 0, 0, 0, 1]}
    encoded_sequence = []

    for nucleotide in seq_string:
        if nucleotide in encoding:
            encoded_sequence.append(encoding[nucleotide])

    return np.vstack(encoded_sequence)

def ordinal_decoder(float_encoded_sequence):
    decoded_seq = ''
    for value in float_encoded_sequence:
        if value == 1.00:
            decoded_seq += 'A'
        elif value == 2.00:
            decoded_seq += 'C'
        elif value == 3.00:
            decoded_seq += 'G'
        elif value == 4.00:
            decoded_seq += 'T'
        else:
            decoded_seq += '' 

    return decoded_seq

def one_hot_decoder(onehot_encoded_sequence):
    mapping = {tuple([1, 0, 0, 0, 0]): 'A', tuple([0, 1, 0, 0, 0]): 'T',
                tuple([0, 0, 1, 0, 0]): 'C', tuple([0, 0, 0, 1, 0]): 'G', tuple([0, 0, 0, 0, 1]): 'N'}
    
    decoded_sequence = ""

    for nucleotide in onehot_encoded_sequence:
        decoded_seq = mapping[tuple(nucleotide.tolist())]
        if decoded_seq != 'N':
            decoded_sequence += decoded_seq

    return decoded_sequence