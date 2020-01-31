import numpy as np
import pandas as pd
#from sklearn.metrics import confusion_matrix
#from matplotlib import pyplot as plt
#from xgboost import XGBClassifier
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#from imblearn.under_sampling import TomekLinks
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
#from sklearn.decomposition import PCA
from collections import Counter
#from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

df_train = pd.read_csv('../../creditcard.csv')
print(df_train.head())

target_count = df_train.Class.value_counts()
#print('Class 0:', target_count[0])
#print('Class 1:', target_count[1])
#print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

#target_count.plot(kind='bar', title='Count (target)')

# Remove 'Time' and 'Class' columns
labels = df_train.columns[1:-1]
print(labels)

X, y = make_classification(n_classes = 2, class_sep = 2,
	weights=[0.4, 0.6], n_informative=2, n_redundant = 1, flip_y = 0,
	n_features = 29, n_clusters_per_class = 1, n_samples = 1000, random_state = 10
)

print(sorted(Counter(y).items()))

smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
print(sorted(Counter(y_resampled).items()))
