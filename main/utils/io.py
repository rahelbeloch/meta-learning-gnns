import datetime

import ujson
import numpy as np


def load_json_file(file_name):
    if file_name.exists():
        return ujson.load(open(file_name, "r"))
    else:
        raise ValueError(f"File {file_name} does not exist!")


def save_json_file(data, file_name, mode="w+", converter=None):
    ujson.dump(data, open(file_name, mode), default=converter)


def create_dir(dir_name):
    if not dir_name.exists():
        dir_name.mkdir(parents=True)
    return dir_name


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
