import spacy
import Levenshtein
import numpy as np
import argparse
from multiprocessing import Process, Queue
from os import path
from utils import basic_utils
import random
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug",
    dest="debug",
    type=bool,
    help="Run in debug mode if true",
    default=True,
)

parser.add_argument(
    "--save_labeled_data",
    dest="save_labeled_data",
    type=bool,
    help="Save labeled data if true",
    default=False,
)

parser.add_argument(
    "--process_data",
    dest="process_data",
    type=bool,
    help="Process input data to labeled data if true",
    default=True,
)

parser.add_argument(
    "--input_label_file",
    dest="input_label_file",
    help="File containing pickled labeled data",
    default=''
)

parser.add_argument(
    "--use_labels",
    dest="use_labels",
    type=bool,
    help="Use handcrafted features",
    default=False,
)
parser.add_argument(
    "--model_type",
    dest="model_type",
    help="Model type used, either rf or lr",
    default='rf'
)
parser.add_argument(
    "--output_dir",
    dest="output_dir",
    help="Directory in which the output is stored",
    default=''
)
parser.add_argument(
    "--labeled_file",
    dest="labeled_file",
    help="Input file with labeled data",
    default='/path/to/repo/data/input_data/word2vec/labeled_polymer_clusters_with_name.tsv'
)

parser.add_argument(
    "--num_processes",
    dest="num_processes",
    help="Number of processes into which we should split the data generation loop",
    type=int,
    default=30
)

args = parser.parse_args()

debug = args.debug
save_labeled_data = args.save_labeled_data
use_labels = args.use_labels
process_data = args.process_data
input_label_file = args.input_label_file
output_dir = args.output_dir
labeled_file = args.labeled_file
model_type = args.model_type
num_processes = args.num_processes


def handcrafted_features(label1, label2, vec1, vec2):
    """Generate a vector of handcrafted features"""
    nlp = spacy.load("en_core_web_sm")
    feature_levenshtein = [Levenshtein.distance(label1, label2)]
    feature_lemma = [int(nlp(label1)[0].lemma_ == nlp(label2)[0].lemma_)]
    feature_cosine_similarity = [np.dot(vec1, vec2)]
    return feature_levenshtein+feature_lemma+feature_cosine_similarity


def pairwise_F1(y_true, y_pred):
    assert len(y_pred)==len(y_true)
    true_pos, false_pos, true_neg, false_neg=0, 0, 0, 0
    # print(true_pos, true_neg, false_neg, false_pos)
    for y_p, y in zip(y_pred, y_true):
        if y==y_p and y==1: true_pos+=1
        elif y==y_p and y==0: true_neg+=1
        elif y!=y_p and y==1: false_neg+=1
        elif y!=y_p and y==0: false_pos+=1
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    F1 = 2*precision*recall/(precision+recall)
    return precision, recall, F1


def read_input_data(labeled_file):
    with open(labeled_file, 'r') as f:
        input_data = f.read()
    input_data = input_data.split('\n')[:-1]
    return input_data


def create_training_data(queue, input_data, start_index, end_index, use_labels=False, debug=False):
    """Take a labeled file as input and produce the labeled dataset with each point as a tuple"""
    training_data = []
    count = 0
    data_size = len(input_data)
    for i in range(start_index, end_index):
        for j in range(i+1, data_size):
            polymer1 = input_data[i]
            polymer2 = input_data[j]
            poly_split_1, poly_split_2 = polymer1.split(' '), polymer2.split(' ')
            pid_1, pid_2 = poly_split_1[0], poly_split_2[0]
            label_1, label_2 = poly_split_1[1], poly_split_2[1]
            poly_vector_1 = [float(d) for d in poly_split_1[3:-1]]
            poly_vector_2 = [float(d) for d in poly_split_2[3:-1]]
            combined_vector = poly_vector_1+poly_vector_2
            if use_labels:
                handcrafted_vector = handcrafted_features(poly_split_1[2], poly_split_2[2], poly_vector_1, poly_vector_2)
                # print(handcrafted_vector)
                combined_vector += handcrafted_vector
            training_data.append(((int(pid_1), int(pid_2)), [combined_vector], int(label_1 == label_2)))
            count += 1
            # print(count)
            if count % 10000 == 0: print(f'count = {count}')
        if debug and count >= 100: break
    queue.put(training_data)
    # return training_data


def dataset_creation(labeled_data):
    """Split labeled data into train and test"""
    one_class = []
    zero_class = []
    for data_point in labeled_data:
        if data_point[2] == 0:
            zero_class.append(data_point)
        else:
            one_class.append(data_point)
    random.shuffle(zero_class)
    random.shuffle(one_class)
    training_data = one_class[:len(one_class) // 2] + zero_class[:len(one_class) // 2]
    test_data = one_class[len(one_class) // 2:] + zero_class[len(one_class) // 2:]
    return training_data, test_data


def process_input_data(input_data):
    queue = Queue()
    # Put in loop where all processes are created. All o/p data in shared variable
    # save the labeled data so created
    data_size = len(input_data)
    processes = []
    labeled_data = []
    i=0
    # Split data into multiple chunks to generate vector representations and process each in a separate process
    for start_index, end_index in chunks(data_size, num_processes):
        print(f'Starting process {i} with {start_index} and {end_index}')
        i+=1
        p = Process(target=create_training_data, args=(queue, input_data, start_index, end_index, use_labels, debug))
        processes.append(p)
        p.start()

    for p in processes:
        print(f'Getting data for process {i}')
        i -= 1
        ret = queue.get()  # will block
        labeled_data.extend(ret)

    for p in processes:
        p.join()
    print(len(labeled_data))
    return labeled_data


def train_model(model_type, training_data):
    _, X, y = zip(*training_data)
    X_array = np.squeeze(np.array(X), axis=1)
    scaler = MinMaxScaler()
    X_array = scaler.fit_transform(X_array)
    if model_type == 'rf':
        trained_model = RandomForestClassifier(max_depth=10, random_state=0)
        trained_model.fit(X_array, y)

    elif model_type == 'lr':
        trained_model = LogisticRegression(random_state=0, max_iter=500)
        trained_model.fit(X_array, y)

    else:
        raise ValueError

    return scaler, trained_model


def test_model(trained_model, scaler, test_data):
    _, X, y = zip(*test_data)
    X = np.squeeze(np.array(X), axis=1)
    X_scale = scaler.transform(X)
    y_pred = trained_model.predict(X_scale)
    return pairwise_F1(y_pred, y)


def chunks(size, n):
    chunk_size = size //n
    for i in range(0, size, chunk_size):
        yield i, min(i+chunk_size-1, size)


def main():
    output_file = path.join(output_dir, 'model_metrics')
    if not path.exists(output_file):
        open(output_file, 'w').close()
    input_data = read_input_data(labeled_file)
    # Split data_size into n start and end indices
    if process_data:
        labeled_data = process_input_data(input_data)
    else:
        with open(path.join(output_dir, input_label_file), 'rb') as f:
            labeled_data = pickle.load(f)

    # Need to store labeled data subject to input switch
    if save_labeled_data:
        time = basic_utils.formatted_time()
        with open(path.join(output_dir, f'{time}_labeled_data.pkl'), 'wb') as f:
            pickle.dump(labeled_data, f)
        with open(path.join(output_dir, f'{time}_labeled_data.info'), 'w') as f:
            f.write(f'This file contains the labeled vector data. use_labels = {use_labels}. The first entry in the tuple is \
                    the 2 pid\'s of polymers followed by the concatenated vector and then a label of 1 or 0 indicating \
                    whether the polymer was same or different')
    training_data, test_data = dataset_creation(labeled_data)
    print(len(training_data))
    print(len(test_data))
    scaler, trained_model = train_model(model_type, training_data)
    prec, recall, F1 = test_model(trained_model, scaler, test_data)
    print('Saving output to file')
    with open(output_file, 'a') as f:
        f.write(f'{basic_utils.formatted_time()} For model {model_type} in debug={debug} mode with use_labels={use_labels} , the performance metrics are:\
                  precision={prec:.3f}, recall={recall:.3f}, F1={F1:.3f}\n\n'
                )


if __name__ == '__main__':
    main()