<h1 align="center">A (More) Realistic Evaluation Setup for Generalisation of Community Models on Malicious Content Detection
</h1>

<!-- TODO add paper URL -->

<!-- TODO add URL to ACL anthology -->

<p align="center">
<a href="https://wandb.ai/verhivo/morph_tag_lemmatize?workspace=user-verhivo">
    <img src="https://img.shields.io/static/v1.svg?logo=arxiv&label=Paper&message=Open%20Paper&color=green"
    alt="Arxiv Link"
    style="float: center;"
    />
</a>
</p>

<p align="center"><b>Ivo Verhoeven<sup>&dagger;</sup>, Pushkar Mishra<sup>&Dagger;</sup>, Rahel Beloch<sup>&dagger;</sup>, Helen Yannakoudakis<sup>&sect;</sup> and Ekaterina Shutova<sup>&dagger;</sup></b></p>

<p align="center"><small>&dagger; ILLC, University of Amsterdam, &Dagger; MetaAI, London, &sect; Dept. of Informatics, King’s College London</small></p>


This is an anonymized version of the codebase accompanying our paper, without any references users or content from the datasets used. As we are bound by the Twitter API terms-of-service, we cannot re-release these datasets.

The environment files used can be found under `./env_cpu.yaml` and `./env_gpu.yaml`. GPU acceleration is strongly recommended.

Four python scripts can be found in `./main` (namely, `preprocess.py`, `train.py`, `evaluate.py`, `transfer.py`). These take the modelling pipeline from preprocessing the data to training models to evaluating models on training data and finally evaluating those models under transfer to a new dataset. Each of these scripts has an accompanying Hydra config file, see `./main/config/*.yaml`.

Three example SLURM job files can be found under `./process_datasets.job`, `./train_protomaml.job` and `./transfer_twitter_sweep.job`. These serve to illustrate how one might interface with the mentioned Python files. These files were actually used for running experiments on [Snellius](https://www.surf.nl/en/services/snellius-the-national-supercomputer); the Dutch national supercomputer.

Please direct your questions to: [ivo.verhoeven@uva.nl](mailto:ivo.verhoeven@uva.nl)

## Structure

```txt
/data/
    various empty folders where the raw data is supposed to be stored
/job_parameters/
    some text files with hyperparameters for the SLURM jobs
/main/
    ├── config
    │       config files needed for Hydra processing, replaces CLI
    ├── data
    │       various empty folders where the processed data should be stored
    ├── data_loading
    │       code for sampling subgraphs and episodes
    ├── data_prep
    │       code taking raw datasets and forming social media graphs
    ├── models
    │       PyTorch models
    ├── utils
    │       various utility functions
    ├── evaluate.py
    │       script for evaluating models after pre-training
    ├── preprocess.py
    │       script taking adjacency lists to meta-learning episode graphs
    ├── train.py
    │       script for pre-training models
    ├── transfer.py
    │       script for transferring to auxilliary datasets
    ├── graph_stats.ipynb
    │       code for computing stats on the sampled episodes
    └── transfer.py
            code for generating tables and plots from results
/results/
    results stored for various model runs
```

## Data

Again, we cannot release the datasets. Instead, we point the user to the repositories of the original datasets, and describe the structure necessary to reconstruct the graphs.

Since we avoid user modelling, it should be possible to reconstruct the graphs using only the edge lists and article content, neither of which are condfidential. However, note that some users or articles have been moderated out, making exact replication of our results impossible.

### GossipCop

*Citation*: Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2020). Fakenewsnet: A data repository with news content, social context, and spatiotemporal information for studying fake news on social media. Big data, 8(3), 171-188.

*Github Repo*: [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)

<details>
<summary>Dataset structure</summary>

```txt
/fake/
    └── /gossipcop-$DOCID/'news content.json'
/real/
    └── /gossipcop-$DOCID/'news content.json'
/retweets/
    └── gossipcop-$USERID.csv
/tweets/
    └── gossipcop-$USERID.csv
/user_followers/
    └── $USERID.json
/user_following/
    └── $USERID.json
```
</details>

### twiterHateSpeech

*Citation*: Waseem, Z., & Hovy, D. (2016, June). Hateful symbols or hateful people? predictive features for hate speech detection on twitter. In Proceedings of the NAACL student research workshop (pp. 88-93).

*Github Repo*: [Hate Speech Twitter annotations](https://github.com/zeeraktalat/hatespeech?tab=readme-ov-file)

<details>
<summary>Dataset structure</summary>
```txt
├── authors.txt
└── twitter_data_waseem_hovy.csv
```
</details>

### CoAID

*Citation*: Cui, L., & Lee, D. (2020). Coaid: Covid-19 healthcare misinformation dataset. arXiv preprint arXiv:2006.00885.

*Github Repo*: [CoAID](https://github.com/cuilimeng/CoAID)

<details>
<summary>Dataset structure</summary>
```txt
/main/
    ├── /05-01-2020/
        ├── NewsFakeCOVID-19.csv
        ├── NewsFakeCOVID-19_tweets.csv
        ├── NewsFakeCOVID-19_replies.csv
        ├── NewsRealCOVID-19.csv
        ├── NewsRealCOVID-19_tweets.csv
        └── NewsRealCOVID-19_replies.csv
    ├── /07-01-2020/
        └── Idem
    ├── /09-01-2020/
        └── Idem
    └── /11-01-2020/
        └── Idem
/retweets/
    └── $USERID.csv
/tweets/
    └── $USERID.csv
/user_followers/
    └── $USERID.json
/user_following/
    └── $USERID.json
```
</details>

## Running Code

We use [Hydra](https://hydra.cc/docs/intro/) as a configuration system. All scripts in `/main/` can be run from the command line, using the Hydra syntax. For example,

```bash
python -u evaluate.py \
    fold=$FOLD \
    data.processed_data_dir=$DATA_DIR \
        structure=episodic_khop \
        learning_algorithm=protomaml \
        ++learning_algorithm.n_inner_updates=10 \
        ++learning_algorithm.lr_inner=5.0e-3 \
        ++learning_algorithm.head_lr_inner=1.0e-2 \
        ++learning_algorithm.reset_classifier=true \
        ++optimizer.lr=5.0e-4 \
        ++optimizer.weight_decay=5.0e-2 \
        ++model.hid_dim=256 \
        ++model.fc_dim=64 \
        ++model.n_heads=3 \
        ++model.node_mask_p=0.10 \
        ++model.dropout=0.50 \
        ++model.attn_dropout=0.10 \
        ++callbacks.early_stopping.metric='val/mcc' \
        ++callbacks.early_stopping.mode=max \
        use_train=false \
        use_val=true \
        use_test=true \
        checkpoint_dir="meta-gnn" \
        checkpoint_name=protomaml
```

would train a ProtoMAML model on an accompanying dataset. See the three `*.job` files for example SLURM bash scripts.

## Results

To reproduce (most of) the figures and tables in the paper, we have included code to parse files in the `/results/` folder. This can be run interactively using the [`./results_parser.ipynb`](./results_parser.ipynb). Be warned, this is extremely poorly formatted code.

## Citation

<!-- TODO add paper URL to citation-->

```bibtex
@inproceedings{verhoeven-etal-2024-generalisation,
    title = "A (More) Realistic Evaluation Setup for Generalisation of Community Models on Malicious Content Detection",
    author = "Verhoeven, Ivo and
        Mishra, Pushkar and
        Beloch, Rahel and
        Yannakoudakis, Helen and
        Shutova, Ekaterina",
    booktitle = "Findings of NAACL 2024",
    year = "2024",
    url = "",
}
```