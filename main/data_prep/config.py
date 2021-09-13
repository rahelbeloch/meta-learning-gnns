RAW_DIR = '../../data/raw/FakeHealth'
TSV_DIR = '../../data/tsv/FakeHealth'
# COMPLETE_DIR = '../../data/complete/FakeHealth'
COMPLETE_DIR = '../data/complete/FakeHealth'

CONTENT_INFO_FILE_NAME = 'docsContentInformation.tsv'

DOC_SPLITS_FILE_NAME = 'docSplits.json'
USER_SPLITS_FILE_NAME = 'userSplits.json'

USER_2_ID_FILE_NAME = 'user2id_lr_top50_train.json'
DOC_2_ID_FILE_NAME = 'doc2id_lr_top50_train.json'
NODE_2_ID_FILE_NAME = 'node2id_lr_top50_train.json'
NODE_TYPE_FILE_NAME = 'node_type_lr_top50_train.npy'

ADJACENCY_MATRIX_FILE = 'adj_matrix_lr_top50_train'
ADJACENCY_MATRIX_FILE_NAME = f'{ADJACENCY_MATRIX_FILE}.npz'

EDGE_TYPE_FILE = 'edge_type_lr_top50'
EDGE_TYPE_FILE_NAME = f'{EDGE_TYPE_FILE}.npz'
EDGE_INDEX_FILE_NAME = EDGE_TYPE_FILE + '_edge.npy'
EDGE_LIST_FILE_NAME = 'edge_list_top50.json'

FEAT_MATRIX_FILE = 'feat_matrix_lr_top50_train'
FEAT_MATRIX_FILE_NAME = f'{FEAT_MATRIX_FILE}.npz'

SPLIT_MASK_FILE_NAME = 'split_mask_lr_30_5.json'

TOP_50_CONSTRAINT = 'top50'
TOP_10_CONSTRAINT = 'top10'

DOC_2_LABELS_FILE_NAME = 'doc2labels_lr_top50_train.json'
LABELS_FILE_NAME = 'labels_list_lr_top50_train.json'
ALL_LABELS_FILE_NAME = 'all_labels_lr_top50_train.json'

DGL_GRAPH_FILE = '%s.dgl'
