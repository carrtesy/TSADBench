import numpy as np
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, roc_auc_score


def get_auroc(gt, anomaly_scores, threshold):
    s = anomaly_scores - threshold
    logit = 1 / (1 + np.exp(-s))  # (N, )
    pred_prob = np.zeros((len(logit), 2))
    pred_prob[:, 0], pred_prob[:, 1] = 1 - logit, logit
    auc = roc_auc_score(gt, anomaly_scores)
    return auc


def get_summary_stats(gt, pred, desc=""):
    '''
    return acc, prec, recall, f1, (tn, fp, fn, tp)
    '''
    acc = accuracy_score(gt, pred)
    p = precision_score(gt, pred, zero_division=1)
    r = recall_score(gt, pred, zero_division=1)
    f1 = f1_score(gt, pred, zero_division=1)
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()

    result = {
        f"Accuracy{desc}": acc,
        f"Precision{desc}": p,
        f"Recall{desc}": r,
        f"F1{desc}": f1,
        f"tn{desc}": tn,
        f"fp{desc}": fp,
        f"fn{desc}": fn,
        f"tp{desc}": tp
    }
    return result


def PA(y, y_pred):
    '''
    Point-Adjust Algorithm
    https://github.com/thuml/Anomaly-Transformer/blob/main/solver.py
    '''
    anomaly_state = False
    y_pred_pa = copy.deepcopy(y_pred)
    for i in range(len(y)):
        if y[i] == 1 and y_pred_pa[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if y[j] == 0:
                    break
                else:
                    if y_pred_pa[j] == 0:
                        y_pred_pa[j] = 1
            for j in range(i, len(y)):
                if y[j] == 0:
                    break
                else:
                    if y_pred_pa[j] == 0:
                        y_pred_pa[j] = 1
        elif y[i] == 0:
            anomaly_state = False
        if anomaly_state:
            y_pred_pa[i] = 1

    return y_pred_pa