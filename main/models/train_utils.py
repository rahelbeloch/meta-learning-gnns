from sklearn.metrics import f1_score


def accuracy(predictions, labels):
    # noinspection PyUnresolvedReferences
    return (labels == predictions.argmax(dim=-1)).float().mean().item()


def evaluation_metrics(predictions, labels):
    pred_cpu = predictions.argmax(dim=-1).detach().cpu()
    labels_cpu = labels.detach().cpu()

    # TODO: set pos label correctly
    f1 = f1_score(labels_cpu, pred_cpu, average='binary', pos_label=1)
    f1_macro = f1_score(labels_cpu, pred_cpu, average='macro')
    f1_micro = f1_score(labels_cpu, pred_cpu, average='micro')
    # recall = recall_score(labels, predictions, average='binary', pos_label=1)
    # precision = precision_score(labels, predictions, average='binary', pos_label=1)
    acc = accuracy(labels, predictions)
    return f1, f1_macro, f1_micro, acc
