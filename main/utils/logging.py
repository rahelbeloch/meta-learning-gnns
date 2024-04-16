import re
import typing
from pathlib import Path

from data_prep.graph_io import GraphIO


def calc_elapsed_time(start, end):
    hours, rem = divmod(end - start, 3600)
    time_hours, time_rem = divmod(end, 3600)
    minutes, seconds = divmod(rem, 60)
    time_mins, _ = divmod(time_rem, 60)
    return int(hours), int(minutes), int(seconds)


def get_results_dir(
    results_dir: str,
    data_args: typing.Dict[str, typing.Any],
    structure_args: typing.Union[typing.Dict[str, typing.Any], str],
    fold: typing.Union[int, str],
    checkpoint: typing.Optional[str] = None,
    version: typing.Optional[str] = None,
):
    # Figure out that dataset dir we're using
    # i.e. the filters and feature extraction methods in place
    graph_io = GraphIO(
        args=data_args,
        version=version,
        delay_making_subdirs=True,
        enforce_raw=False,
    )

    dataset_str = f"dataset[{graph_io.dataset}]_{graph_io.data_dir_name}"

    # Figure out the structure of that dataset and thus model
    if isinstance(structure_args, str):
        if checkpoint is not None:
            raise ValueError(
                "Checkpoint should not be specified for the baseline models."
            )

        if structure_args == "text" or structure_args == "text_baseline":
            structure_str = "text_baseline"
            checkpoint = "text_baseline"

        elif structure_args == "social" or structure_args == "social_baseline":
            structure_str = "social_baseline"
            checkpoint = "social_baseline"

    else:
        if checkpoint is None:
            raise ValueError("Checkpoint should be specified for the baseline models.")

        structure_str = f"structure[{structure_args['structure']}]_mode[{structure_args['structure_mode']}]_k[{structure_args['labels_per_graph']}]"

    # Figure out which specific checkpoint to use
    if isinstance(fold, int):
        fold_str = f"fold[{fold}]"
        checkpoint_str = f"checkpoint[{checkpoint}]"
    elif isinstance(fold, str) and fold == "summary":
        fold_str = "summary"
        checkpoint_str = ""
    else:
        raise ValueError("Fold must either be an `int` or the `str` 'summary'.")

    return Path(results_dir) / dataset_str / structure_str / fold_str / checkpoint_str


def get_config_from_results_dir(
    results_dir: typing.Union[Path, str], skip_summary: bool = True
):
    results_config = dict()
    bracktets_pattern = r"\[(.*?)\]"

    if not skip_summary:
        raise NotImplementedError("Haven't thought about how to deal with summaries")

    if isinstance(results_dir, str):
        results_dir = Path(results_dir)

    parts = results_dir.parts[2:]

    dataset_str = parts[0]
    dataset_str_parts = re.split(r"\_(?![^[]*])", dataset_str)

    results_config["dataset"] = re.search(
        bracktets_pattern, dataset_str_parts[0]
    ).group(1)
    results_config["seed"] = int(
        re.search(bracktets_pattern, dataset_str_parts[1]).group(1)
    )
    results_config["splits"] = int(
        re.search(bracktets_pattern, dataset_str_parts[2]).group(1)
    )
    results_config["minlen"] = int(
        re.search(bracktets_pattern, dataset_str_parts[3]).group(1)
    )
    results_config["filter_isolated_users"] = bool(
        re.search(bracktets_pattern, dataset_str_parts[4]).group(1)
    )
    results_config["top_users"] = int(
        re.search(bracktets_pattern, dataset_str_parts[5]).group(1)
    )
    results_config["top_users_excluded"] = int(
        re.search(bracktets_pattern, dataset_str_parts[6]).group(1)
    )
    results_config["userdoc"] = int(
        re.search(bracktets_pattern, dataset_str_parts[7]).group(1)
    )
    results_config["featuretype"] = re.search(
        bracktets_pattern, dataset_str_parts[8]
    ).group(1)

    vocab_str = re.findall(bracktets_pattern, dataset_str_parts[9])
    results_config["vocab_origin"] = vocab_str[0]
    results_config["compression"] = vocab_str[1]

    compresion_str = vocab_str[2].split("x")
    if compresion_str[0] == "None":
        results_config["vocab_size"] = None
        results_config["compressed_size"] = None
    else:
        results_config["vocab_size"] = int(compresion_str[0])
        results_config["compressed_size"] = int(compresion_str[1])

    user_features_str = re.findall(bracktets_pattern, dataset_str_parts[10])
    results_config["user_compression_pre_or_post"] = user_features_str[0]
    results_config["user2doc_aggregator"] = user_features_str[1]

    if "version" in dataset_str:
        results_config["version"] = re.search(
            bracktets_pattern, dataset_str_parts[-1]
        ).group(1)
    else:
        results_config["version"] = None

    fold_str = parts[2]
    if fold_str == "summary":
        return None
    else:
        results_config["fold"] = re.search(bracktets_pattern, fold_str).group(1)

    structure_str = parts[1]
    if structure_str == "text_baseline" or structure_str == "social_baseline":
        results_config["checkpoint"] = structure_str
        results_config["structure"] = "full"
        results_config["structure_mode"] = "inductive"
        results_config["k"] = 4

    else:
        structure_str_parts = re.split(r"\_(?![^[]*])", structure_str)

        results_config["structure"] = re.search(
            bracktets_pattern, structure_str_parts[0]
        ).group(1)
        results_config["structure_mode"] = re.search(
            bracktets_pattern, structure_str_parts[1]
        ).group(1)
        results_config["k"] = re.search(
            bracktets_pattern, structure_str_parts[2]
        ).group(1)

        checkpoint_str = parts[3]
        results_config["checkpoint"] = re.search(
            bracktets_pattern, checkpoint_str
        ).group(1)

    return results_config
