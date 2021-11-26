import json
import os
import re
from string import punctuation

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOP_WORDS = set(open(os.path.join(BASE_DIR, '../resources/stopwords.txt'), 'r').read().split())


def split_data(splits, size, data):
    assert splits == 1, "Currently not supporting more than 1 test split!"

    x_data, y_data, doc_names = data

    sss = StratifiedShuffleSplit(n_splits=splits, test_size=size, random_state=21)

    # USING the real texts for the splits creates issues, because there are different doc IDs with same texts!
    # split1_index, split2_index = next(sss.split(x_data, y_data))

    split1_index, split2_index = next(sss.split(np.zeros(y_data.shape), y_data))

    split1 = (x_data[split1_index], y_data[split1_index], doc_names[split1_index])
    split2 = (x_data[split2_index], y_data[split2_index], doc_names[split2_index])

    return split1, split2


def get_label_distribution(labels):
    fake, real = (labels == 1).sum(), (labels == 0).sum()
    denom = fake + real
    return fake / denom, real / denom


def print_label_distribution(labels):
    fake, real = get_label_distribution(labels)
    print(f"\nFake labels in train split  = {fake * 100:.2f} %")
    print(f"Real labels in train split  = {real * 100:.2f} %")


def load_json_file(file_name):
    if file_name.exists():
        return json.load(open(file_name, 'r'))
    else:
        raise ValueError(f"File {file_name} does not exist!")


def save_json_file(data, file_name, mode='w+', converter=None):
    json.dump(data, open(file_name, mode), default=converter)


def calc_elapsed_time(start, end):
    hours, rem = divmod(end - start, 3600)
    time_hours, time_rem = divmod(end, 3600)
    minutes, seconds = divmod(rem, 60)
    time_mins, _ = divmod(time_rem, 60)
    return int(hours), int(minutes), int(seconds)


def sanitize_text(text):
    text = text.lower()

    # replace special symbols in the text
    text = re.sub('[' + punctuation + ']', ' ', text)
    text = re.sub('\\b[0-9]+\\b', '', text)
    text = re.sub('\\s+', ' ', text)

    # text = re.sub(r'#[\w-]+', 'hashtag', text)
    # text = re.sub(r'https?://\S+', 'url', text)

    # filter out stopwords
    text = [w for w in text.split(' ') if w not in STOP_WORDS]

    return ' '.join(text)
