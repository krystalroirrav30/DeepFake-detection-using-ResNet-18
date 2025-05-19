from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

def compute_metrics(y_true, y_pred_probs):
    y_pred = (y_pred_probs > 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_pred_probs),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }
