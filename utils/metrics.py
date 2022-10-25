import numpy as np
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def get_statistics(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    a, p, r, f = accuracy_score(y, y_pred),\
              precision_score(y, y_pred, zero_division=1), \
              recall_score(y, y_pred, zero_division=1), \
              f1_score(y, y_pred, zero_division=1)
    return cm, a, p, r, f


def latency(y, y_pred):
    pass

