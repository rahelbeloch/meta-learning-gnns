import os
import re
import pickle
from string import punctuation
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoConfig, AutoTokenizer

from utils.io import load_json_file


def sanitize_text(
    text, remove_punctuation: bool = False, replace_with_string: bool = False
):
    text = re.sub("\\s+", " ", text)

    # (twitter hate speech) preprocessing
    text = re.sub("\\b[0-9]+\\b", "", text)

    # (gossipcop) preprocessing from safer paper
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"#[\w-]+", "hashtag" if replace_with_string else " ", text)
    text = re.sub(r"@[\w-]+", "user" if replace_with_string else " ", text)
    text = re.sub(r"https?://\S+", "URL" if replace_with_string else " ", text)

    if remove_punctuation:
        text = text.encode("ascii", errors="ignore").strip().decode("ascii")
        text = re.sub("[" + punctuation + "]", " ", text)

    return text


class OneHotTokenizer(object):
    def __init__(
        self,
        vocab_size: int,
        oov_idx: int = -1,
        stop_words_fp: str = "./stopwords.txt",
        use_joint_vocab: bool = True,
    ):
        self.vocab_size = vocab_size

        self.oov_idx = oov_idx

        self.stop_words_fp = stop_words_fp
        if self.stop_words_fp is not None:
            with open(self.stop_words_fp, "r") as f:
                self.stop_words = f.read().split()
        else:
            print("\n>>>WARNING: KEEPING STOP WORDS<<<\n")

        try:
            word_tokenize("test string")
        except LookupError:
            print("\n>>>WARNING: DOWNLOADING PUNKT TOKENIZER FROM NLTK<<<\n")
            nltk.download("punkt")

        self.use_joint_vocab = use_joint_vocab

    def preprocess_string(self, text: str):
        text = text.lower()

        text = sanitize_text(text, remove_punctuation=True)

        text = word_tokenize(text)

        # filter out stopwords
        if self.stop_words_fp is not None:
            text = {w for w in text if w not in self.stop_words}

        return text

    def build_vocab(self, all_texts):
        all_texts = list(map(self.preprocess_string, all_texts))

        word_freq = Counter(token for doc in all_texts for token in doc)

        self.vocab = {
            word: i
            for i, (word, _) in enumerate(word_freq.most_common(self.vocab_size))
        }

    def stoi(self, token: str):
        return self.vocab.get(token, self.oov_idx)

    def __call__(self, text):
        sanitized_text = self.preprocess_string(text)

        indices = set()
        oov_count = 0
        for token in sanitized_text:
            idx = self.stoi(token)

            if idx == self.oov_idx:
                oov_count += 1
                continue

            indices.add(idx)

        return {
            "input_ids": list(indices),
            "oov_count": oov_count,
            "length": len(indices),
        }

    def save(self, fp):
        state_dict = {
            "vocab_size": self.vocab_size,
            "oov_idx": self.oov_idx,
            "stop_words_fp": self.stop_words_fp,
            "vocab": self.vocab,
        }

        with open(fp, "wb") as f:
            pickle.dump(state_dict, f)

    @classmethod
    def load(cls, fp):
        with open(fp, "rb") as f:
            state_dict = pickle.load(f)

        instance = cls(
            vocab_size=state_dict["vocab_size"],
            oov_idx=state_dict["oov_idx"],
            stop_words_fp=state_dict["stop_words_fp"],
        )
        instance.vocab = state_dict["vocab"]

        return instance


class LMTokenizer(object):
    def __init__(self, lm_name, **kwargs):
        self.lm_name = lm_name

        self.config = AutoConfig.from_pretrained(
            self.lm_name,
            cache_dir="./resources",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.lm_name,
            use_fast=True,
            cache_dir="./resources",
        )
        self.tokenizer_kwargs = kwargs

    def __call__(self, text: str):
        sanitized_text = sanitize_text(text, replace_with_string=True)

        # The -2 is necessary to account for start and end tokens
        # Some models add these above maximum allowed length
        # Assuming this has minimal effect
        return self.tokenizer(
            sanitized_text,
            max_length=self.config.max_position_embeddings - 2,
            **self.tokenizer_kwargs,
        )

    def save(self, fp):
        state_dict = {
            "lm_name": self.lm_name,
            "tokenizer_kwargs": self.tokenizer_kwargs,
        }

        with open(fp, "wb") as f:
            pickle.dump(state_dict, f)

    @classmethod
    def load(cls, fp):
        with open(fp, "rb") as f:
            state_dict = pickle.load(f)

        instance = cls(
            lm_name=state_dict["lm_name"],
            **state_dict["tokenizer_kwargs"],
        )

        return instance
