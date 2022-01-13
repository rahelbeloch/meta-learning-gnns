RAW_DIR = 'raw'
TSV_DIR = 'tsv'
COMPLETE_DIR = 'complete'

num_doc_nodes = '5000'
COMPLETE_small_DIR = 'complete-%s' % num_doc_nodes
TSV_small_DIR = 'tsv-%s' % num_doc_nodes

# REQUIRED FILES

CONTENT_INFO_FILE_NAME = 'docsContentInformation.tsv'

DOC_SPLITS_FOLDER_NAME = 'splits-%s-%s-train%s-val%s-test%s'
DOC_SPLITS_FILE_NAME = 'doc-splits-%s-%s-train%s-val%s-test%s.json'
USER_SPLITS_FILE_NAME = 'user-splits-%s-%s-train%s-val%s-test%s.json'

VALID_USERS = 'valid_users.json'
RESTRICTED_USERS = 'restricted_users_%s.json'
TOP_K_USERS_EXCLUDED = 'top_users_excluded_%s.json'

FEAT_MATRIX_FILE = 'feat_matrix_train_type=%s_vsize=%s_train=%s_val=%s_test=%s'
FEAT_MATRIX_FILE_NAME = f'{FEAT_MATRIX_FILE}.npz'

ALL_LABELS_FILE_NAME = 'labels_train_val_test_type=%s_vsize=%s_train=%s_val=%s_test=%s.json'

EDGE_LIST_FILE_NAME = 'edge_list_type=%s_vsize=%s_train=%s_val=%s_test=%s.json'
EDGE_TYPE_FILE = 'edge_type_type=%s_vsize=%s_train=%s_val=%s_test=%s'
EDGE_TYPE_FILE_NAME = f'{EDGE_TYPE_FILE}.npz'

SPLIT_MASK_FILE_NAME = 'split_mask_type=%s_vsize=%s_train=%s_val=%s_test=%s.json'

USER_2_ID_FILE_NAME = 'user2id_train_type=%s_vsize=%s_train=%s_val=%s_test=%s.json'
DOC_2_ID_FILE_NAME = 'doc2id_train_type=%s_vsize=%s_train=%s_val=%s_test=%s.json'
DOC_2_LABELS_FILE_NAME = 'doc2labels.json'

# NOT REQUIRED

# NODE_2_ID_FILE_NAME = 'node2id_train.json'
# NODE_TYPE_FILE_NAME = 'node_type_train.npy'

ADJACENCY_MATRIX_FILE = 'adj_matrix_train'
ADJACENCY_MATRIX_FILE_NAME = f'{ADJACENCY_MATRIX_FILE}.npz'

# EDGE_INDEX_FILE_NAME = EDGE_TYPE_FILE + '_edge.npy'


# TRAIN_LABELS_FILE_NAME = 'labels_train_val_type=%s_vsize=%s_train=%s_val=%s_test=%s.json'
