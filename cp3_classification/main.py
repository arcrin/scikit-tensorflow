#%% Imports
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


#%% Fetch and load mnist
mnist = fetch_openml("mnist_784", version=1)

#%% Separate data and label
x, y = mnist["data"], mnist["target"]

#%% Plot an image
some_digit = x[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()


#%% Convert label to unsigned integer
y = y.astype(np.uint8)

#%% Separate training and test set
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

#%% Binary classifier, 5-detector
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#%% SGD Classifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)


#%% Performance, Measuring Accuracy using Cross-Validation
skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(x_train, y_train_5):
    clone_clf = clone(sgd_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train_5[train_index]
    x_test_fold = x_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))


#%%
cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring="accuracy")

#%% Dumb classifier

class Never5Classifier(BaseEstimator):
    def fit(self, x, y=None):
        pass

    def predict(self, x):
        return np.zeros((len(x), 1), dtype=bool)

#%%
never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, x_train, y_train_5, cv=3, scoring="accuracy"))


#%% Confusion Matrix
y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3)
res = confusion_matrix(y_train_5, y_train_pred)

#%%
y_train_perfect_predictions = y_train_5
res = confusion_matrix(y_train_5, y_train_perfect_predictions)

#%% Precision and Recall
precision_score_res = precision_score(y_train_5, y_train_pred)
recall_score_res = recall_score(y_train_5, y_train_pred)

#%% F1 score
f1_score_res = f1_score(y_train_5, y_train_pred)

#%% Precision/Recall Trade off
y_scores = sgd_clf.decision_function([some_digit])
threshold = 0
y_some_digit_pred = (y_scores > threshold)

#%%
threshold = 8000
y_some_digit_pred = (y_scores > threshold)

#%% Picking threshold
y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3, method="decision_function")

#%% Precision Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


#%%
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend()

#%%
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
y_train_pred_90 = (y_scores >= threshold_90_precision)


#%%
precision_score(y_train_5, y_train_pred_90)

#%%
recall_score(y_train_5, y_train_pred_90)

#%% The ROC (receiver operating characteristic) Curve
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    if label is not None:
        plt.legend()

#%%
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
plot_roc_curve(fpr, tpr)

#%% ROC AUC
roc_auc_score(y_train_5, y_scores)


#%% RnadomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv=3, method="predict_proba")
