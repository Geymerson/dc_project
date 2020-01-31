import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
warnings.filterwarnings("ignore", category=UserWarning)

rng = np.random.RandomState(2)
h = .02  # step size in the mesh
df = pd.read_csv('creditcard.csv')

cls = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

target = ['Class']

plt.style.use('ggplot')

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]


def metrics(estimator, x, y):
    y_pred = estimator.predict(x)

    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, pos_label=3, average='macro')
    recall = recall_score(y, y_pred, pos_label=3, average='macro')
    f1 = f1_score(y, y_pred, average='macro')
    roc = roc_auc_score(y, y_pred)

    print(acc, ',', precision, ',', recall, ',', f1, ',', roc)
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

X = df[cls]
y = df[target]

cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Correlation with output variable
cor_target = abs(cor["Class"])
# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.5]
print(relevant_features)

# Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(X)
# Fitting sm.OLS model
model = sm.OLS(y, X_1).fit()

print(model.pvalues)

# Backward Elimination
cols = list(X.columns)
pmax = 1

while len(cols) > 0:
    p = []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y, X_1).fit()
    p = pd.Series(model.pvalues.values[1:], index=cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if pmax > 0.05:
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

X = df[selected_features_BE]

df.Class.hist()
plt.show()

print(df.Class.value_counts())

skip = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
        "Neural Net"]
# iterate over classifiers
for name, clf in zip(names, classifiers):
    if name in skip:
        continue
    skf = StratifiedKFold(n_splits=10)
    print('Accuracy,Precision,Recall,F1,ROC')
    print(name, cross_val_score(clf, X, y.values.ravel(), cv=skf, scoring=metrics))
