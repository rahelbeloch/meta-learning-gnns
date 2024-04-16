import os

import hydra
from omegaconf import OmegaConf

from data_prep.content_processing import (
    GossipcopContentProcessor,
    TwitterHateSpeechContentProcessor,
    CoaidContentProcessor,
    HealthStoryContentProcessor,
)
from data_prep.graph_processing import (
    GossipcopGraphProcessor,
    TwitterHateSpeechGraphProcessor,
    CoaidGraphProcessor,
    HealthStoryGraphProcessor,
)
from data_prep.post_processing import (
    train_social_baseline,
    train_feature_extractor_and_compress,
)
from data_loading import get_dataset

os.environ["HYDRA_FULL_ERROR"] = "1"


def preprocess_gossipcop(args):
    if not args["skip_data_processing"]:
        content_processor = GossipcopContentProcessor(
            args["data"],
            version=args["version"],
            overwrite=args["data"]["overwrite"],
        )
        content_processor.reset()
        content_processor.prep()
        doc_dataset = content_processor.sanitize_documents()
        content_processor.filter_users()
        content_processor.apply_filters(doc_dataset)

        del doc_dataset, content_processor

    else:
        print("Skipping content processing.")

    if not args["skip_graph_processing"]:
        graph_processor = GossipcopGraphProcessor(
            args["data"],
            version=args["version"],
            overwrite=args["data"]["overwrite"],
        )
        graph_processor.generate_node_id_mappings()
        graph_processor.generate_adjacency_matrix()
        graph_processor.split_documents()

        del graph_processor

    else:
        print("Skipping graph processing.")


def preprocess_twitterhatespeech(args):
    if not args["skip_data_processing"]:
        content_processor = TwitterHateSpeechContentProcessor(
            args["data"],
            content_file=args["data"]["content_file"],
            version=args["version"],
            overwrite=args["data"]["overwrite"],
        )
        content_processor.reset()
        content_processor.prep()
        doc_dataset = content_processor.sanitize_documents()
        content_processor.filter_users()
        content_processor.apply_filters(doc_dataset)

        del doc_dataset, content_processor

    else:
        print("Skipping content processing.")

    if not args["skip_graph_processing"]:
        graph_processor = TwitterHateSpeechGraphProcessor(
            args["data"],
            version=args["version"],
            overwrite=args["data"]["overwrite"],
        )
        graph_processor.generate_node_id_mappings()
        graph_processor.generate_adjacency_matrix()
        graph_processor.split_documents()

        del graph_processor

    else:
        print("Skipping graph processing.")


def preprocess_coaid(args):
    if not args["skip_data_processing"]:
        content_processor = CoaidContentProcessor(
            args["data"],
            version=args["version"],
            overwrite=args["data"]["overwrite"],
        )
        content_processor.reset()
        content_processor.prep()
        doc_dataset = content_processor.sanitize_documents()
        content_processor.filter_users()
        content_processor.apply_filters(doc_dataset)

        del doc_dataset, content_processor

    else:
        print("Skipping content processing.")

    if not args["skip_graph_processing"]:
        graph_processor = CoaidGraphProcessor(
            args["data"],
            version=args["version"],
            overwrite=args["data"]["overwrite"],
        )
        graph_processor.generate_node_id_mappings()
        graph_processor.generate_adjacency_matrix()
        graph_processor.split_documents()

        del graph_processor

    else:
        print("Skipping graph processing.")


def preprocess_healthstory(args):
    if not args["skip_data_processing"]:
        content_processor = HealthStoryContentProcessor(
            args["data"],
            version=args["version"],
            overwrite=args["data"]["overwrite"],
        )
        content_processor.reset()
        content_processor.prep()
        doc_dataset = content_processor.sanitize_documents()
        content_processor.filter_users()
        content_processor.apply_filters(doc_dataset)

        del doc_dataset, content_processor

    else:
        print("Skipping content processing.")

    if not args["skip_graph_processing"]:
        graph_processor = HealthStoryGraphProcessor(
            args["data"],
            version=args["version"],
            overwrite=args["data"]["overwrite"],
        )
        graph_processor.generate_node_id_mappings()
        graph_processor.generate_adjacency_matrix()
        graph_processor.split_documents()

        del graph_processor

    else:
        print("Skipping graph processing.")


@hydra.main(version_base=None, config_path="./config", config_name="preprocess")
def main(args):
    # ==========================================================================
    # Argument processing
    # ==========================================================================
    args = OmegaConf.to_container(args, resolve=True)
    if args["print_config"]:
        print(OmegaConf.to_yaml(args))
    else:
        "Config loaded but not printing."

    # ==========================================================================
    # Content and graph preprocessing
    # ==========================================================================
    if args["data"]["dataset"] == "gossipcop":
        preprocess_gossipcop(args)

    elif args["data"]["dataset"] == "twitterHateSpeech":
        preprocess_twitterhatespeech(args)

    elif args["data"]["dataset"] == "CoAID":
        preprocess_coaid(args)

    elif args["data"]["dataset"] == "HealthStory":
        preprocess_healthstory(args)

    else:
        raise ValueError(
            f"Somehow, the dataset_name is not recognized: {args['dataset']}"
        )

    # ==========================================================================
    # Feature extraction
    # ==========================================================================
    if not args["skip_feature_extraction"]:
        if args["data"]["num_splits"] > 0:
            train_social_baseline(args=args, version=args["version"])

        train_feature_extractor_and_compress(args=args, version=args["version"])

    else:
        print("Skipping feature extraction.")

    # ==========================================================================
    # Structuring
    # ==========================================================================
    if not args["skip_structure"]:
        for split in ["test"]:
            graph_dataset = get_dataset(args, split, load=False)

            graph_dataset.prep()
            graph_dataset.build_graph()
            graph_dataset.split_graph()

            if callable(getattr(graph_dataset, "generate_batches", None)):
                graph_dataset.partition_into_batches()
                graph_dataset.generate_batches(
                    num_workers=args["structure"]["num_workers"]
                )

            graph_dataset.save()

    else:
        print("Skipping structuring.")


if __name__ == "__main__":
    main()
