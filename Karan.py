# %%
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import statistics
import matplotlib.pyplot as plt
from sklearn import preprocessing
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# %%
class BinaryAndMulticlassClassifier(BaseEstimator, TransformerMixin):
    def __init__(self, binary_classifier, multiclass_classifier):
        self.binary_classifier = binary_classifier
        self.multiclass_classifier = multiclass_classifier

    def fit(self, X, y):
        # Binary classification fit
        self.binary_classifier.fit(X, np.where(y == 3, 1, 0))
        
        # Apply binary classifier
        binary_predictions = self.binary_classifier.predict(X)
        
        # Prepare data for multiclass classification
        X_not_class_3 = X[binary_predictions == 0]
        y_not_class_3 = y[binary_predictions == 0]
        
        # Multiclass classification fit
        self.multiclass_classifier.fit(X_not_class_3, y_not_class_3)
        return self

    def predict(self, X):
        binary_predictions = self.binary_classifier.predict(X)
        X_not_class_3 = X[binary_predictions == 0]
        
        multiclass_predictions = np.full(X.shape[0], -1)  # Initialize with default class
        multiclass_predictions[binary_predictions == 0] = self.multiclass_classifier.predict(X_not_class_3)
        return np.where(binary_predictions == 1, 3, multiclass_predictions)


# %%
df = pd.read_csv("Encoded_data.csv")

# %%
df.drop("Unnamed: 0", axis = 1, inplace = True)

# %%
df["status"].value_counts()

# %%
X = df.drop(['status'], axis=1)
y = df['status']

# %%
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

print(f"Original dataset shape: {Counter(y)}")
print(f"Resampled dataset shape: {Counter(y_res)}")

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
y_train_binary = np.where(y_train == 3, 1, 0)

# %%
binary_pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    #('pca', PCA(n_components=10)),
    ('binary_classifier', LogisticRegression(penalty='l1', solver='saga', max_iter=50, multi_class='multinomial', C=1))
])

# %%
binary_pipeline.fit(X_train, y_train_binary)

# %%
binary_predictions = binary_pipeline.predict(X_test)

# %%
X_test_not_class_3 = X_test[binary_predictions == 0]
y_test_not_class_3 = y_test[binary_predictions == 0]

# %%
X_train_not_class_3 = X_train[y_train != 3]
y_train_not_class_3 = y_train[y_train != 3]

# %%
multiclass_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    #('pca', PCA(n_components=10)),
    ('multiclass_classifier', RandomForestClassifier(random_state=42))
])


# %%
multiclass_pipeline.fit(X_train_not_class_3, y_train_not_class_3)

# %%
combined_pipeline = Pipeline([
    ('model', BinaryAndMulticlassClassifier(binary_pipeline, multiclass_pipeline))
])

# Fit the combined pipeline
combined_pipeline.fit(X_train, y_train)

# Predict with the combined pipeline
final_predictions = combined_pipeline.predict(X_test)
print(f"Final predictions: {Counter(final_predictions)}")

# %%
#multiclass_predictions = multiclass_pipeline.predict(X_test_not_class_3)

# %%
#final_predictions = binary_predictions.copy()

# %%
#final_predictions[binary_predictions == 0] = multiclass_predictions

# %%
#print("Final predictions:", final_predictions)

# %%
accuracy_score(y_test, final_predictions)

# %%
import dill

with open('combined_plzz.pkl', 'wb') as file:
    dill.dump(combined_pipeline, file)

# %%
import joblib

# Save the model
joblib.dump(combined_pipeline, 'combined_yr.pkl')

# %%



