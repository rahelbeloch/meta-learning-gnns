import abc
import datetime

import nltk
from importlib_resources import files
from torchtext.vocab import GloVe

nltk.download('punkt')

from data_prep.data_preprocess_utils import *

from data_prep.config import *

FEATURE_TYPES = ['one-hot', 'glove-sum', 'glove-average']

# -1 because index 0 is not free in the Glove vocabulary
NIV_IDX = (-1, 'NIV')


class GraphIO:

    def __init__(self, dataset, feature_type, max_vocab, data_dir, tsv_dir=TSV_DIR, complete_dir=COMPLETE_DIR):
        self.dataset = dataset

        data_path = files(data_dir)
        raw_path = data_path / RAW_DIR
        if not (raw_path / dataset).exists():
            raise ValueError(f"Wanting to preprocess data for dataset '{dataset}', but raw data in path"
                             f" with raw data '{raw_path / dataset}' does not exist!")

        self.data_raw_dir = raw_path
        self.data_tsv_dir = self.create_dir(data_path / tsv_dir / self.dataset).parent
        self.data_complete_dir = self.create_dir(data_path / complete_dir / self.dataset).parent

        self.non_interaction_docs, self.max_vocab = None, max_vocab

        if feature_type not in FEATURE_TYPES:
            raise ValueError(f"Trying to create features of type {feature_type} which is not supported!")
        self.feature_type = feature_type

    def print_step(self, step_title):
        print(f'\n{"-" * 100}\n \t\t\t {step_title} for {self.dataset} dataset.\n{"-" * 100}')

    @staticmethod
    def load_if_exists(file_name):
        if file_name.exists():
            return load_json_file(file_name)
        else:
            raise ValueError(f"Wanting to load file with name {file_name}, but this file does not exist!!")

    @staticmethod
    def create_dir(dir_name):
        if not dir_name.exists():
            dir_name.mkdir(parents=True)
        return dir_name

    def data_raw_path(self, *parts):
        return self.data_raw_dir.joinpath(*parts)

    def data_tsv_path(self, *parts):
        return self.data_tsv_dir.joinpath(self.dataset, *parts)

    def data_complete_path(self, *parts):
        return self.data_complete_dir.joinpath(self.dataset, *parts)

    @staticmethod
    def np_converter(obj):
        """
        A converter which can be used when dumping JSON strings to files, as JSON can not work with numpy data types.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()

    def get_vocab_token2idx(self, texts):
        if self.feature_type == 'one-hot':
            # create own vocabulary from all data
            vocab = self.build_vocab(texts)
            return vocab, len(vocab)
        elif 'glove' in self.feature_type:
            feature_size = 200
            glove = GloVe(name='twitter.27B', dim=feature_size, max_vectors=self.max_vocab)
            # check if words are in the inflected form
            return glove, feature_size
        else:
            raise ValueError(f"Trying to create features of type {self.feature_type} which is not unknown!")

    @staticmethod
    def as_vocab_indices(vocabulary, tokens):
        return [vocabulary[token] if token in vocabulary else vocabulary[NIV_IDX[1]] for token in tokens]

    def build_vocab(self, all_text_tokens, max_count=-1, min_count=2):

        # creating word frequency dict
        word_freq = {}
        for doc_key, tokens in all_text_tokens.items():
            for token in set(tokens):
                word_freq[token] = word_freq.get(token, 0) + 1
        word_freq = [(f, w) for (w, f) in word_freq.items()]
        word_freq.sort(reverse=True)

        # collect token counts
        token_counts = []
        for (count, token) in word_freq:
            if max_count != -1 and count > max_count:
                continue
            if count < min_count:
                continue
            token_counts.append((count, token))

        token_counts.sort(reverse=True)
        if self.max_vocab != -1:
            token_counts = token_counts[:self.max_vocab]
        # NIV: not in vocab token, i.e., out of vocab
        token_counts.append(NIV_IDX)

        vocab = {}
        for (i, (count, token)) in enumerate(token_counts):
            vocab[token] = i + 1

        return vocab

    @property
    @abc.abstractmethod
    def labels(self):
        raise NotImplementedError
