"""Helper functions for data pre-processing"""

from dataclasses import dataclass, asdict
from typing import Optional, Union, List
import re
import collections
import logging
import tensorflow as tf
import pandas as pd
import json
import os
import glob
import numpy as np

logger = logging.getLogger(__name__)


# Class for triletter vocab dictionary
class L3G:
    dict = {}
    invdict = {}

    def __init__(self, l3g_path):
        with open(l3g_path, 'r', encoding='utf-8') as fp:
            i = 1  # note that the dictionary now increases all trigram index by 1!!!
            while True:
                s = fp.readline().strip('\n\r')
                if s == '':
                    break
                self.dict[s] = i
                self.invdict[i] = s
                i += 1
        return


# Triletter Encoder Class
class TriLetterExtractor:
    def __init__(self, l3g_path, dim=49292):
        self.l3ginst = L3G(l3g_path)
        self.dimension = dim
        self.content = []
        self.n_seq = 0
        self.invalid = re.compile('[^a-zA-Z0-9 ]')
        self.multispace = re.compile('  +')
        self.max_token_num = 12

    def extract_features(self, qstr, max_n_letters=20, max_seq_len=12):
        qseq, qmask = self.extract_from_sentence(qstr, max_n_letters, max_seq_len)  # add word index if needed
        return qseq, qmask

    def extract_from_words(self, words, max_n_letters=20, max_seq_len=12):
        valid_words = []
        for _, word in enumerate(words):
            word = self.invalid.sub('', word)
            word = word.strip()
            if word != '':
                valid_words.append(word)
        return self._from_words_to_id_sequence(valid_words, max_n_letters, max_seq_len)

    def extract_from_sentence(self, text, max_n_letters=20, max_seq_len=12):
        step1 = text.lower()
        step2 = self.invalid.sub('', step1)
        step3 = self.multispace.sub(' ', step2)
        step4 = step3.strip()
        words = step4.split(' ')
        return self._from_words_to_id_sequence(words, max_n_letters, max_seq_len)

    def _from_words_to_id_sequence(self, words, max_n_letters=20, max_seq_len=12):
        n_seq = min(len(words), max_seq_len)
        n_letter = max_n_letters
        feature_seq = [0] * (max_seq_len * max_n_letters)
        seq_mask = [0] * max_seq_len
        for i in range(n_seq):
            if words[i] == '':
                words[i] = '#'
            word = '#' + words[i] + '#'
            n_letter = min(len(word) - 2, max_n_letters)
            for j in range(n_letter):
                s = word[j:(j + 3)]
                if s in self.l3ginst.dict:
                    feature_seq[i * max_n_letters + j] = self.l3ginst.dict[s]
            seq_mask[i] = 1
        return feature_seq, seq_mask


@dataclass
class InputExample:
    text_a: str
    text_b: Optional[str] = None
    label: Optional[Union[int, float]] = None
    text_a_neighbors: Optional[List[str]] = None
    text_b_neighbors: Optional[List[str]] = None
    text_a_neighbors_impression: Optional[List[int]] = None
    text_b_neighbors_impression: Optional[List[int]] = None
    text_a_neighbors_click: Optional[List[int]] = None
    text_b_neighbors_click: Optional[List[int]] = None

    def to_json_string(self):
        return json.dumps(asdict(self), indent=2) + "\n"


@dataclass
class InputFeatures:
    input_ids_a: List[int]
    input_ids_b: List[int]

    attention_mask_a: Optional[List[int]] = None
    attention_mask_b: Optional[List[int]] = None

    input_ids_a_neighbor: Optional[List[List[int]]] = None
    input_ids_b_neighbor: Optional[List[List[int]]] = None

    attention_mask_a_neighbor: Optional[List[List[int]]] = None
    attention_mask_b_neighbor: Optional[List[List[int]]] = None

    impression_a_neighbor: Optional[List[int]] = None
    impression_b_neighbor: Optional[List[int]] = None

    click_a_neighbor: Optional[List[int]] = None
    click_b_neighbor: Optional[List[int]] = None

    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        return json.dumps(asdict(self)) + "\n"


# Read a single pandas row into a InputExample
def get_example_from_row(row, a_fanouts=[], b_fanouts=[], int_label=True, read_label=True, read_neigh_weights=True):
    text_a_neighbors = None
    text_b_neighbors = None
    text_a_neighbors_impression = None
    text_b_neighbors_impression = None
    text_a_neighbors_click = None
    text_b_neighbors_click = None

    if len(a_fanouts) > 0 and a_fanouts[0] > 0:
        layer_node = 1
        text_a_neighbors = []
        if read_neigh_weights:
            text_a_neighbors_impression = []
            text_a_neighbors_click = []

        for layer in range(len(a_fanouts)):
            layer_node *= a_fanouts[layer]
            neighbors = row['Q_Neighbor_' + str(layer)].split('!!@@##$$')
            if read_neigh_weights:
                neighbors_impression = 0 if neighbors[0] == "" else list(map(int, row['Q_Neighbor_Impression_' + str(layer)].split('!!@@##$$')))
                neighbors_click = 0 if neighbors[0] == "" else list(map(int, row['Q_Neighbor_Click_' + str(layer)].split('!!@@##$$')))

            if len(neighbors) == 1 and neighbors[0] == "":
                text_a_neighbors += ["[PAD]"] * layer_node
                if read_neigh_weights:
                    text_a_neighbors_impression += [1] * layer_node
                    text_a_neighbors_click += [0] * layer_node
            else:
                neigh_length = min(len(neighbors), layer_node)
                text_a_neighbors += neighbors[:neigh_length] + ["[PAD]"] * (layer_node - neigh_length)
                if read_neigh_weights:
                    text_a_neighbors_impression += neighbors_impression[:neigh_length] + [1] * (layer_node - neigh_length)
                    text_a_neighbors_click += neighbors_click[:neigh_length] + [0] * (layer_node - neigh_length)

    if len(b_fanouts) > 0 and b_fanouts[0] > 0:
        layer_node = 1
        text_b_neighbors = []
        if read_neigh_weights:
            text_b_neighbors_impression = []
            text_b_neighbors_click = []

        for layer in range(len(b_fanouts)):
            layer_node *= b_fanouts[layer]
            neighbors = row['K_Neighbor_' + str(layer)].split('!!@@##$$')
            if read_neigh_weights:
                neighbors_impression = 0 if neighbors[0] == "" else list(map(int, row['K_Neighbor_Impression_' + str(layer)].split('!!@@##$$')))
                neighbors_click = 0 if neighbors[0] == "" else list(map(int, row['K_Neighbor_Click_' + str(layer)].split('!!@@##$$')))

            if len(neighbors) == 1 and neighbors[0] == "":
                text_b_neighbors += ["[PAD]"] * layer_node
                if read_neigh_weights:
                    text_b_neighbors_impression += [1] * layer_node
                    text_b_neighbors_click += [0] * layer_node
            else:
                neigh_length = min(len(neighbors), layer_node)
                text_b_neighbors += neighbors[:neigh_length] + ["[PAD]"] * (layer_node - neigh_length)
                if read_neigh_weights:
                    text_b_neighbors_impression += neighbors_impression[:neigh_length] + [1] * (layer_node - neigh_length)
                    text_b_neighbors_click += neighbors_click[:neigh_length] + [0] * (layer_node - neigh_length)

    return InputExample(
        text_a=row['Query'],
        text_b=row['Keyword'],
        label=(row['QK_Rel'] if int_label else row['RoBERTaScore']) if read_label else None,
        text_a_neighbors=text_a_neighbors,
        text_b_neighbors=text_b_neighbors,
        text_a_neighbors_impression=text_a_neighbors_impression,
        text_b_neighbors_impression=text_b_neighbors_impression,
        text_a_neighbors_click=text_a_neighbors_click,
        text_b_neighbors_click=text_b_neighbors_click
    )


# Convert pandas DataFrame into a list of InputExamples
def get_examples_from_pd(data, int_label=True, a_fanouts=[], b_fanouts=[]):
    return [get_example_from_row(data.iloc[i], a_fanouts=a_fanouts, b_fanouts=b_fanouts, int_label=int_label) for i in
            range(len(data))]


# Convert a list of InputExamples into a list of InputFeatures
def convert_examples_to_features(examples, extractor, max_seq_len=12, max_n_letters=20, a_fanouts=[], b_fanouts=[]):
    def label_from_example(ex: InputExample) -> Union[int, float, None]:
        label = ex.label
        return None if (label is None or label < 0) else label

    features = []

    if type(extractor) == TriLetterExtractor:
        for example in examples:
            input_ids_a, attention_mask_a = extractor.extract_features(example.text_a, max_n_letters=max_n_letters,
                                                                       max_seq_len=max_seq_len)
            input_ids_b, attention_mask_b = extractor.extract_features(example.text_b, max_n_letters=max_n_letters,
                                                                       max_seq_len=max_seq_len)
            input_ids_a_neighbor = None
            input_ids_b_neighbor = None
            attention_mask_a_neighbor = None
            attention_mask_b_neighbor = None
            impression_a_neighbor = None
            impression_b_neighbor = None
            click_a_neighbor = None
            click_b_neighbor = None

            if len(a_fanouts) > 0 and a_fanouts[0] > 0:
                input_ids_a_neighbor = []
                attention_mask_a_neighbor = []
                impression_a_neighbor = example.text_a_neighbors_impression
                click_a_neighbor = example.text_a_neighbors_click

                for text in example.text_a_neighbors:
                    input_ids, attention_mask = extractor.extract_features(text, max_n_letters=max_n_letters,
                                                                           max_seq_len=max_seq_len)
                    input_ids_a_neighbor.append(input_ids)
                    attention_mask_a_neighbor.append(attention_mask)

            if len(b_fanouts) > 0 and b_fanouts[0] > 0:
                input_ids_b_neighbor = []
                attention_mask_b_neighbor = []
                impression_b_neighbor = example.text_b_neighbors_impression
                click_b_neighbor = example.text_b_neighbors_click

                for text in example.text_b_neighbors:
                    input_ids, attention_mask = extractor.extract_features(text, max_n_letters=max_n_letters,
                                                                           max_seq_len=max_seq_len)
                    input_ids_b_neighbor.append(input_ids)
                    attention_mask_b_neighbor.append(attention_mask)

            feature = {
                'input_ids_a': input_ids_a,
                'attention_mask_a': attention_mask_a,
                'input_ids_b': input_ids_b,
                'attention_mask_b': attention_mask_b,
                'label': label_from_example(example),
                'input_ids_a_neighbor': input_ids_a_neighbor,
                'attention_mask_a_neighbor': attention_mask_a_neighbor,
                'input_ids_b_neighbor': input_ids_b_neighbor,
                'attention_mask_b_neighbor': attention_mask_b_neighbor,
                'impression_a_neighbor': impression_a_neighbor,
                'impression_b_neighbor': impression_b_neighbor,
                'click_a_neighbor': click_a_neighbor,
                'click_b_neighbor': click_b_neighbor
            }
            features.append(InputFeatures(**feature))
    else:
        labels = [label_from_example(example) for example in examples]
        text_a_list = [example.text_a for example in examples]
        text_b_list = [example.text_b for example in examples]

        batch_encoding_a = extractor(text_a_list, max_length=max_seq_len, pad_to_max_length=True,
                                     return_token_type_ids=False, truncation=True)
        batch_encoding_b = extractor(text_b_list, max_length=max_seq_len, pad_to_max_length=True,
                                     return_token_type_ids=False, truncation=True)

        if len(a_fanouts) > 0 and a_fanouts[0] > 0:
            a_neighbors = [extractor(example.text_a_neighbors, max_length=max_seq_len, pad_to_max_length=True,
                                     return_token_type_ids=False, truncation=True) for example in examples]
        if len(b_fanouts) > 0 and b_fanouts[0] > 0:
            b_neighbors = [extractor(example.text_b_neighbors, max_length=max_seq_len, pad_to_max_length=True,
                                     return_token_type_ids=False, truncation=True) for example in examples]

        for i in range(len(text_a_list)):
            inputs = {k + "_a": batch_encoding_a[k][i] for k in batch_encoding_a}
            inputs.update({k + "_b": batch_encoding_b[k][i] for k in batch_encoding_b})
            if len(a_fanouts) > 0 and a_fanouts[0] > 0:
                inputs.update({k + "_a_neighbor": a_neighbors[i][k] for k in a_neighbors[i]})
                inputs.update({"impression_a_neighbor": examples[i].text_a_neighbors_impression,
                               "click_a_neighbor": examples[i].text_a_neighbors_click})
            if len(b_fanouts) > 0 and b_fanouts[0] > 0:
                inputs.update({k + "_b_neighbor": b_neighbors[i][k] for k in b_neighbors[i]})
                inputs.update({"impression_b_neighbor": examples[i].text_b_neighbors_impression,
                               "click_b_neighbor": examples[i].text_b_neighbors_click})
            feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)

    return features


# Pandas parsing functions to prevent data error
def convert_int(x):
    try:
        return int(x)
    except Exception as e:
        print(e)
        return -99


def convert_float(x):
    try:
        return float(x)
    except Exception as e:
        print(e)
        return -99.0


def convert_float_to_int(x):
    try:
        y = float(x)
        return 0 if y < 0.5 else 1
    except Exception as e:
        print(e)
        return -99


# Helper function to create tf dataset features
def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f


def feature_to_dict(f, int_label=True, a_fanouts=[], b_fanouts=[]):
    f_s = {}
    f_s["input_ids_a_0"] = tf.convert_to_tensor(np.array(f.input_ids_a), dtype=tf.int32)[tf.newaxis,:]
    f_s["input_ids_b_0"] = tf.convert_to_tensor(np.array(f.input_ids_a), dtype=tf.int32)[tf.newaxis,:]
    f_s["attention_mask_a_0"] = tf.convert_to_tensor(np.array(f.input_ids_a), dtype=tf.int32)[tf.newaxis,:]
    f_s["attention_mask_b_0"] = tf.convert_to_tensor(np.array(f.input_ids_a), dtype=tf.int32)[tf.newaxis,:]


    if len(a_fanouts) > 0 and a_fanouts[0] > 0:
        layer_node = 1
        for layer in range(len(a_fanouts)):
            layer_node *= a_fanouts[layer]
            f_s['input_ids_a_' + str(layer+1)] = tf.convert_to_tensor(np.array(f.input_ids_a_neighbor), dtype=tf.int32)[tf.newaxis,:,:]
            f_s['attention_mask_a_' + str(layer+1)] = tf.convert_to_tensor(np.array(f.attention_mask_a_neighbor), dtype=tf.int32)[tf.newaxis,:,:]
    if len(b_fanouts) > 0 and b_fanouts[0] > 0:
        layer_node = 1
        for layer in range(len(b_fanouts)):
            layer_node *= b_fanouts[layer]
            f_s['input_ids_b_' + str(layer+1)] = tf.convert_to_tensor(np.array(f.input_ids_b_neighbor), dtype=tf.int32)[tf.newaxis,:,:]
            f_s['attention_mask_b_' + str(layer+1)] = tf.convert_to_tensor(np.array(f.attention_mask_b_neighbor), dtype=tf.int32)[tf.newaxis,:,:]
    return f_s

# Read raw tsv file by chunks and save to tf record files
def process_datasets_to_file(data_path: str, extractor, write_filename: str, max_seq_len=12, max_n_letters=20,
                             int_label=True, chunksize=1e6, top=None, convert_to_int=False, a_fanouts=[], b_fanouts=[], skip_chunk=0, n_chunk=0):
    names = ['Query', 'Keyword', 'QK_Rel' if (int_label or convert_to_int) else 'RoBERTaScore']
    converters = {"Query": str, "Keyword": str}

    if int_label:
        converters["QK_Rel"] = convert_int
    elif convert_to_int:
        converters["QK_Rel"] = convert_float_to_int
    else:
        converters["RoBERTaScore"] = convert_float

    if len(a_fanouts) > 0 and a_fanouts[0] > 0:
        layer_node = 1
        for layer in range(len(a_fanouts)):
            names.append('Q_Neighbor_' + str(layer))
            converters['Q_Neighbor_' + str(layer)] = str
            names.append('Q_Neighbor_Impression_' + str(layer))
            converters['Q_Neighbor_Impression_' + str(layer)] = str
            names.append('Q_Neighbor_Click_' + str(layer))
            converters['Q_Neighbor_Click_' + str(layer)] = str

    names.append('Q_Dist')
    converters['Q_Dist'] = convert_int

    if len(b_fanouts) > 0 and b_fanouts[0] > 0:
        layer_node = 1
        for layer in range(len(b_fanouts)):
            names.append('K_Neighbor_' + str(layer))
            converters['K_Neighbor_' + str(layer)] = str
            names.append('K_Neighbor_Impression_' + str(layer))
            converters['K_Neighbor_Impression_' + str(layer)] = str
            names.append('K_Neighbor_Click_' + str(layer))
            converters['K_Neighbor_Click_' + str(layer)] = str

    names.append('K_Dist')
    converters['K_Dist'] = convert_int

    names.append('rand')
    names.append('rank')
    converters['rand'] = convert_float
    converters['rank'] = convert_int

    logger.info(names)
    logger.info(converters)

    int_label = int_label or convert_to_int

    chunk_n = skip_chunk

    count = 0
    for chunk in pd.read_csv(data_path, sep='\t', header=None, names=names, error_bad_lines=False,
                             converters=converters, chunksize=chunksize, skiprows=int(skip_chunk * chunksize)):
        writer = tf.io.TFRecordWriter(os.path.join(write_filename, 'data_%d.tf_record' % chunk_n))
        if int_label:
            chunk = chunk.loc[chunk["QK_Rel"] >= 0]
        else:
            chunk = chunk.loc[chunk["RoBERTaScore"] >= 0.0]

        logger.info("Process chunk %d" % chunk_n)

        logger.info(len(chunk))
        examples = get_examples_from_pd(chunk, int_label=int_label, a_fanouts=a_fanouts, b_fanouts=b_fanouts)
        logger.info("Finish loading examples")
        features = convert_examples_to_features(examples, extractor, max_n_letters=max_n_letters,
                                                max_seq_len=max_seq_len, a_fanouts=a_fanouts, b_fanouts=b_fanouts)
        logger.info("Finish converting features")

        for i, f in enumerate(features):
            logger.info(count)
            count += 1
            f_s = collections.OrderedDict()
            f_s["input_ids_a"] = create_int_feature(f.input_ids_a)
            f_s["input_ids_b"] = create_int_feature(f.input_ids_b)
            f_s["attention_mask_a"] = create_int_feature(f.attention_mask_a)
            f_s["attention_mask_b"] = create_int_feature(f.attention_mask_b)
            f_s["label"] = create_int_feature([f.label]) if int_label else create_float_feature([f.label])

            if len(a_fanouts) > 0 and a_fanouts[0] > 0:
                layer_node = 1
                n = 0
                for layer in range(len(a_fanouts)):
                    layer_node *= a_fanouts[layer]
                    for k in range(layer_node):
                        f_s['input_ids_a_' + str(layer) + '_' + str(k)] = create_int_feature(f.input_ids_a_neighbor[n])
                        f_s['attention_mask_a_' + str(layer) + '_' + str(k)] = create_int_feature(f.attention_mask_a_neighbor[n])
                        f_s['impression_a_neighbor_' + str(layer) + '_' + str(k)] = create_int_feature([f.impression_a_neighbor[n]])
                        f_s['click_a_neighbor_' + str(layer) + '_' + str(k)] = create_int_feature(
                            [f.click_a_neighbor[n]])
                        n += 1
            if len(b_fanouts) > 0 and b_fanouts[0] > 0:
                layer_node = 1
                n = 0
                for layer in range(len(b_fanouts)):
                    layer_node *= b_fanouts[layer]
                    for k in range(layer_node):
                        f_s['input_ids_b_' + str(layer) + '_' + str(k)] = create_int_feature(f.input_ids_b_neighbor[n])
                        f_s['attention_mask_b_' + str(layer) + '_' + str(k)] = create_int_feature(f.attention_mask_b_neighbor[n])
                        f_s['impression_b_neighbor_' + str(layer) + '_' + str(k)] = create_int_feature(
                            [f.impression_b_neighbor[n]])
                        f_s['click_b_neighbor_' + str(layer) + '_' + str(k)] = create_int_feature(
                            [f.click_b_neighbor[n]])
                        n += 1

            tf_example = tf.train.Example(features=tf.train.Features(feature=f_s))
            writer.write(tf_example.SerializeToString())
        chunk_n += 1
        if chunk_n - skip_chunk == n_chunk:
            break

        if top and count >= top:
            break

        writer.close()


# Read and parse tf record dataset into desired input format
def read_preprocessed_datasets(filepath: str, max_seq_len=16, max_n_letters=20, is_triletter=False, int_label=True, a_fanouts=[], b_fanouts=[], neigh_weights=False):
    filenames = glob.glob(os.path.join(filepath, '*.tf_record'))

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    filenames.sort(key=natural_keys)
    dataset = tf.data.TFRecordDataset(filenames)
    length = max_n_letters * max_seq_len if is_triletter else max_seq_len

    def _decode_record(record):
        name_to_features = {"input_ids_a": tf.io.FixedLenFeature([length], tf.int64),
                            "input_ids_b": tf.io.FixedLenFeature([length], tf.int64),
                            "attention_mask_a": tf.io.FixedLenFeature([max_seq_len], tf.int64),
                            "attention_mask_b": tf.io.FixedLenFeature([max_seq_len], tf.int64),
                            "label": tf.io.FixedLenFeature([], tf.int64) if int_label else tf.io.FixedLenFeature([],
                                                                                                                 tf.float32)
                            }

        if len(a_fanouts) > 0 and a_fanouts[0] > 0:
            layer_node = 1
            for layer in range(len(a_fanouts)):
                layer_node *= a_fanouts[layer]
                for k in range(layer_node):
                    name_to_features['input_ids_a_' + str(layer) + '_' + str(k)] = tf.io.FixedLenFeature([length], tf.int64)
                    name_to_features['attention_mask_a_' + str(layer) + '_' + str(k)] = tf.io.FixedLenFeature([max_seq_len], tf.int64)
                    if neigh_weights:
                        name_to_features['impression_a_neighbor_' + str(layer) + '_' + str(k)] = tf.io.FixedLenFeature([], tf.int64)
                        name_to_features['click_a_neighbor_' + str(layer) + '_' + str(k)] = tf.io.FixedLenFeature([], tf.int64)
        if len(b_fanouts) > 0 and b_fanouts[0] > 0:
            layer_node = 1
            for layer in range(len(b_fanouts)):
                layer_node *= b_fanouts[layer]
                for k in range(layer_node):
                    name_to_features['input_ids_b_' + str(layer) + '_' + str(k)] = tf.io.FixedLenFeature([length], tf.int64)
                    name_to_features['attention_mask_b_' + str(layer) + '_' + str(k)] = tf.io.FixedLenFeature([max_seq_len], tf.int64)
                    if neigh_weights:
                        name_to_features['impression_b_neighbor_' + str(layer) + '_' + str(k)] = tf.io.FixedLenFeature([], tf.int64)
                        name_to_features['click_b_neighbor_' + str(layer) + '_' + str(k)] = tf.io.FixedLenFeature([], tf.int64)
        example = tf.io.parse_single_example(record, name_to_features)

        for name in list(example.keys()):
            if name != "label" or int_label:
                example[name] = tf.cast(example[name], tf.int32)

        input_ids_a = example["input_ids_a"]
        attention_mask_a = example["attention_mask_a"]
        input_ids_b = example["input_ids_b"]
        attention_mask_b = example["attention_mask_b"]

        return_dict = {
            "input_ids_a_0": input_ids_a,
            "input_ids_b_0": input_ids_b,
            "attention_mask_a_0": attention_mask_a,
            "attention_mask_b_0": attention_mask_b,
        }

        if len(a_fanouts) > 0 and a_fanouts[0] > 0:
            layer_node = 1
            for layer in range(len(a_fanouts)):
                layer_node *= a_fanouts[layer]
                tmp_ids = []
                tmp_mask = []
                tmp_impression = []
                tmp_click = []

                for i in range(layer_node):
                    tmp_ids.append(example['input_ids_a_' + str(layer) + '_' + str(i)])
                    tmp_mask.append(example['attention_mask_a_' + str(layer) + '_' + str(i)])
                    if neigh_weights:
                        tmp_impression.append(example['impression_a_neighbor_' + str(layer) + '_' + str(i)])
                        tmp_click.append(example['click_a_neighbor_' + str(layer) + '_' + str(i)])
                return_dict['input_ids_a_' + str(layer+1)] = tf.stack(tmp_ids)
                return_dict['attention_mask_a_' + str(layer+1)] = tf.stack(tmp_mask)
                if neigh_weights:
                    return_dict['impression_a_' + str(layer+1)] = tf.stack(tmp_impression)
                    return_dict['click_a_' + str(layer+1)] = tf.stack(tmp_click)

        if len(b_fanouts) > 0 and b_fanouts[0] > 0:
            layer_node = 1
            for layer in range(len(b_fanouts)):
                layer_node *= b_fanouts[layer]
                tmp_ids = []
                tmp_mask = []
                tmp_impression = []
                tmp_click = []

                for i in range(layer_node):
                    tmp_ids.append(example['input_ids_b_' + str(layer) + '_' + str(i)])
                    tmp_mask.append(example['attention_mask_b_' + str(layer) + '_' + str(i)])
                    if neigh_weights:
                        tmp_impression.append(example['impression_b_neighbor_' + str(layer) + '_' + str(i)])
                        tmp_click.append(example['click_b_neighbor_' + str(layer) + '_' + str(i)])
                return_dict['input_ids_b_' + str(layer+1)] = tf.stack(tmp_ids)
                return_dict['attention_mask_b_' + str(layer+1)] = tf.stack(tmp_mask)
                if neigh_weights:
                    return_dict['impression_b_' + str(layer+1)] = tf.stack(tmp_impression)
                    return_dict['click_b_' + str(layer+1)] = tf.stack(tmp_click)
        return (
            return_dict,
            example["label"],
        )

    return dataset.map(_decode_record)