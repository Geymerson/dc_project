import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

rng = np.random.RandomState(2)
h = .02  # step size in the mesh
df = pd.read_csv('creditcard.csv')

cls = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

target = ['Class']

plt.style.use('ggplot')

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]


def metrics(estimator, x, y):
    y_pred = estimator.predict(x)

    acc = accuracy_score(y, y_pred),
    precision = precision_score(y, y_pred, pos_label=3, average='macro'),
    recall = recall_score(y, y_pred, pos_label=3, average='macro'),
    f1 = f1_score(y, y_pred, average='macro')
    roc = roc_auc_score(y, y_pred)

    print('Accuracy', acc)
    print('Precision', precision)
    print('Recall', recall)
    print('F1 score', f1)
    print('ROC', roc)

    return acc + precision + recall + f1 + roc


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)

plt.show()

df.Class.hist()
plt.show()

print(df.Class.value_counts())
X = df[cls]
y = df[target]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    skf = StratifiedKFold(n_splits=10)
    print(name, cross_val_score(clf, X, y.values.ravel(), cv=skf, scoring=metrics))
