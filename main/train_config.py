META_MODELS = ['gmeta', 'proto-maml', 'maml']
SUPPORTED_MODELS = ['gat', 'prototypical'] + META_MODELS

# SHOTS = [5, 10, 20, 40]
SHOTS = [4, 8, 12, 16]

GOSSIPCOP_QUERY_SAMPLES = {'train': {0: 1640, 1: 820}, 'val': {0: 200, 1: 50}, 'test': {0: 480, 1: 120}}
GOSSIPCOP_EPISODES = {
    'train': {4: 820, 8: 410, 12: 205, 16: 205},
    'val': {4: 50, 8: 25, 12: 25, 16: 25},
    'test': {4: 120, 8: 60, 12: 40, 16: 30}
}

# we are not training on twitter, therefore using all samples for test!
TWITTER_QUERY_SAMPLES = {'test': {0: 912, 1: 1824, 2: 3648}}
TWITTER_EPISODES = {'test': {4: 912, 8: 456, 12: 304, 16: 228}}

LOG_PATH = "../logs/"
