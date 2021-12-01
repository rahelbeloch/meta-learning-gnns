import sklearn


def accuracy(predictions, labels):
    # noinspection PyUnresolvedReferences
    return (labels == predictions.argmax(dim=-1)).float().mean().item()


def f1(predictions, targets, average='binary'):
    predictions_cpu = predictions.argmax(dim=-1).detach().cpu()
    targets_cpu = targets.detach().cpu()
    return sklearn.metrics.f1_score(targets_cpu, predictions_cpu, average=average).item()
