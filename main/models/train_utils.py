from sklearn.metrics import f1_score


def accuracy(predictions, labels):
    # noinspection PyUnresolvedReferences
    return (labels == predictions.argmax(dim=-1)).float().mean().item()


def evaluation_metrics(predictions, labels, f1_target_label):
    pred_cpu = predictions.argmax(dim=-1).detach().cpu()
    labels_cpu = labels.detach().cpu()

    print(pred_cpu)
    print(labels_cpu)

    # F1 score of the target class (fake for gossipcop and racism for twitter)
    f1 = None
    if f1_target_label is not None and f1_target_label in labels:
        f1 = f1_score(labels_cpu, pred_cpu, average='binary', pos_label=f1_target_label)

    f1_macro = f1_score(labels_cpu, pred_cpu, average='macro')
    f1_micro = f1_score(labels_cpu, pred_cpu, average='micro')
    # recall = recall_score(labels, predictions, average='binary', pos_label=1)
    # precision = precision_score(labels, predictions, average='binary', pos_label=1)

    return f1, f1_macro, f1_micro, accuracy(pred_cpu, labels_cpu)
