from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from scipy.stats import wasserstein_distance


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average="weighted")


def accuracy_per_class(preds, labels, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        tqdm.write(f"Class: {label_dict_inverse[label]}")
        tqdm.write(f"Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n")


def plot_confusion_matrix(preds, labels, label_dict):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    label_to_name = {value: key for key, value in label_dict.items()}

    # Вычисляем confusion matrix
    cm = confusion_matrix(labels_flat, preds_flat)
    label_names = list(label_dict.keys())  # ['Easy', 'Normal']
    # label_names = ['Very Easy', 'Easy', 'Normal', 'Hard', 'Very Hard']#[label_to_name.get(label, "Unknown") for label in np.unique(labels_flat)]

    # Визуализация confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm[::-1],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names[::-1],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Истинные классы")
    plt.xlabel("Предсказанные классы")
    plt.show()


def metrics_per_class(preds, labels, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    # Calculate confusion matrix
    cm = confusion_matrix(labels_flat, preds_flat)

    for label in np.unique(labels_flat):
        # True Positives
        TP = cm[label, label]
        # False Positives: sum of the corresponding column minus TP
        FP = np.sum(cm[:, label]) - TP
        # False Negatives: sum of the corresponding row minus TP
        FN = np.sum(cm[label, :]) - TP

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0
        )

        tqdm.write(f"Class: {label_dict_inverse[label]}")
        tqdm.write(f"Accuracy: {TP}/{TP+FN} (True Positives / Total Actual Positives)")
        tqdm.write(f"Precision: {precision:.2f}")
        tqdm.write(f"Recall: {recall:.2f}")
        tqdm.write(f"F1 Score: {f1:.2f}\n")


def pearson_corr(x, y):
    return (torch.dot(x, y) / (torch.norm(x) * torch.norm(y))).item()


def ANC(L1, L2):
    X = L1 - L1.mean(dim=0, keepdim=True)
    Y = L2 - L2.mean(dim=0, keepdim=True)

    anc = 0
    for i in range(X.size()[0]):
        anc += abs(pearson_corr(X[i], Y[i]))

    return anc / X.size()[0]


def count_ANC(layers1, layers2):
    anc = []
    for i in range(13):
        anc.append(ANC(layers1[i], layers2[i]))

    return anc


def wasserstein(matrix1, matrix2):
    vector1 = matrix1.cpu().numpy().flatten()
    vector2 = matrix2.cpu().numpy().flatten()

    distance = wasserstein_distance(vector1, vector2)
    return distance


def count_wasserstein_distance(layers1, layers2):
    dist = []
    for i in range(13):
        dist.append(wasserstein(layers1[i], layers2[i]))

    return dist
