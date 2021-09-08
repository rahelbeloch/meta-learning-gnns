import abc


class GraphDataset():
    """
    Parent class for graph datasets. It is required to implement a preprocess and generate_features methods.
    """

    def __init__(self, corpus, config):
        super().__init__()

        self.config = config
        self.corpus = corpus

        # self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # self._num_classes = corpus.num_classes
        #
        # train_texts, train_labels = corpus.train_data
        # val_texts, val_labels = corpus.val_data
        # test_texts, test_labels = corpus.test_data
        #
        # self._raw_texts = train_texts + val_texts + test_texts
        #
        # # Hopefully no tokenizer makes a token "doc.i"
        # all_docs = ['doc.{}'.format(i) for i in range(len(self._raw_texts))]
        #
        # print('Preprocess corpus')
        # lower_threshold, upper_threshold = (10, 30) if type(corpus) == IMDbData else (4, 50)
        # tokenized_text, self._tokens = self._preprocess(lower_threshold, upper_threshold)
        #
        # iton = list(all_docs + self._tokens)
        # ntoi = {iton[i]: i for i in range(len(iton))}
        # self.num_nodes = len(iton)
        #
        # print('Compute tf.idf')
        # tf_idf, tf_idf_words = tf_idf_mtx(tokenized_text)
        #
        # print('Compute PMI scores')
        # pmi_score = get_PMI(tokenized_text)
        #
        # print('Generate edges')
        # edge_index, edge_attr = self._generate_edges(tf_idf, tf_idf_words, pmi_score, ntoi)
        #
        # print('Generate masks')
        # train_mask, val_mask, test_mask = self._generate_masks(len(train_texts), len(val_texts), len(test_texts))
        #
        # doc_labels = train_labels + val_labels + test_labels
        #
        # self._labels = torch.full((len(iton),), -1, device=self._device)
        # self._labels[:len(doc_labels)] = torch.tensor(doc_labels)
        #
        # self._data = Data(edge_index=edge_index, edge_attr=edge_attr, y=self._labels, num_nodes=self.num_nodes)
        # self._data.doc_features, self._data.word_features = self._generate_features()
        # self._data.train_mask = train_mask
        # self._data.val_mask = val_mask
        # self._data.test_mask = test_mask

    def labels(self):
        """
        Return the labels of data points.

        Returns:
            labels (Tensor): Document and word labels
        """
        return self._labels

    # def as_dataloader(self):
    #     """
    #     Return this dataset as data loader from torch geometric.
    #     """
    #     return geom_data.DataLoader(self)

    @staticmethod
    def _generate_edges(tf_idf, tf_idf_words, pmi_scores, ntoi):
        """
        Generates edge list and weights based on tf.idf and PMI.
        Args:
            tf_idf (SparseMatrix): sklearn Sparse matrix object containing tf.idf values.
            tf_idf_words (list): List of words according to the tf.idf matrix.
            pmi_scores (dict): Dictionary of word pairs and corresponding PMI scores.
            ntoi (dict): Dictionary mapping from nodes to indices.
        Returns:
            edge_index (Tensor): List of edges.
            edge_attr (Tensor): List of edge weights.
        """
        edge_index = []
        edge_attr = []

        # Document-word edges
        for d_ind, doc in enumerate(tf_idf):
            for tf_idf_ind in doc.indices:
                # Convert index from tf.idf to index in ntoi
                word = tf_idf_words[tf_idf_ind]
                w_ind = ntoi[word]

                edge_index.append([d_ind, w_ind])
                edge_index.append([w_ind, d_ind])
                edge_attr.append(tf_idf[d_ind, tf_idf_ind])
                edge_attr.append(tf_idf[d_ind, tf_idf_ind])

        # Word-word edges
        for (word_i, word_j), score in pmi_scores.items():
            w_i_ind = ntoi[word_i]
            w_j_ind = ntoi[word_j]
            edge_index.append([w_i_ind, w_j_ind])
            edge_index.append([w_j_ind, w_i_ind])
            edge_attr.append(score)
            edge_attr.append(score)

        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr).float()
        return edge_index, edge_attr

    def _generate_masks(self, train_num, val_num, test_num):
        """
        Generates masking for the different splits in the dataset.
        Args:
            train_num (int): Number of training documents.
            val_num (int): Number of validation documents.
            test_num (int): Number of test documents.
            all_num (int): Number of all nodes, including words
        Returns:
            train_mask (Tensor): Training mask as boolean tensor.
            val_mask (Tensor): Validation mask as boolean tensor.
            test_mask (Tensor): Test mask as boolean tensor.
        """
        train_mask = torch.zeros(self.num_nodes, device=self._device)
        train_mask[:train_num] = 1

        val_mask = torch.zeros(self.num_nodes, device=self._device)
        val_mask[train_num:train_num + val_num] = 1

        # Mask all non-test docs
        test_mask = torch.zeros(self.num_nodes, device=self._device)
        test_mask[val_num + train_num:val_num + train_num + test_num] = 1

        return train_mask.bool(), val_mask.bool(), test_mask.bool()

    def _preprocess(self, lower_threshold=4, upper_threshold=50):
        """
        Preprocesses the corpus.
        Returns:
            tokenized_text (List): List of tokenized documents texts.
            tokens (List): List of all tokens.
        """

    @abc.abstractmethod
    def _generate_features(self):
        """
        Generates node features.
        """
        raise NotImplementedError

    def len(self):
        return 1

    def get(self, idx):
        return self._data
