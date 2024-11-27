import numpy as np
from sklearn.metrics import confusion_matrix


def f1_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def roc_auc_score(y_true, y_pred):
    # Ordena as predicções e rótulos verdadeiros
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true = np.array(y_true)[sorted_indices]
    y_pred = np.array(y_pred)[sorted_indices]
    
    # Calcula as taxas de verdadeiros positivos e falsos positivos
    tpr = np.cumsum(y_true) / np.sum(y_true)
    fpr = np.cumsum(1 - y_true) / np.sum(1 - y_true)
    
    # Calcula a área sob a curva ROC
    auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1])) / 2
    return auc