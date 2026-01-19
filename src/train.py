import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_recall_curve, auc
from scipy.stats import uniform, randint
import os
import joblib
from datetime import datetime

df=pd.read_csv('data/creditcard.csv')


X=df.drop(columns=['Class'])
y=df['Class']
#Using stratify to keep the same ratio of classes in train, validation, and test set
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)
X_train_resampled, y_train_resampled=SMOTE(random_state=42).fit_resample(X_train, y_train)


param_dist = {
    'n_estimators': randint(1000, 3001),
    'learning_rate': uniform(0.001, 0.199),
    'max_depth': randint(3,21),
    'subsample': uniform(0.5, 0.5),
    'reg_lambda': uniform(0.0, 1.0)
}

XGB=XGBClassifier(random_state=42)
XGB_random = RandomizedSearchCV(estimator=XGB, param_distributions=param_dist, n_iter=60, cv=3, verbose=2, random_state=42, n_jobs=-1, scoring='average_precision')
XGB_random.fit(X_train_resampled, y_train_resampled)
y_score=XGB_random.predict_proba(X_test)[:,1]
precision, recall, _ = precision_recall_curve(y_test, y_score)
print(auc(recall, precision))

os.makedirs('models', exist_ok=True)
model_filename = 'models/xgb_model.pkl'
joblib.dump(XGB_random, model_filename)

print(f"Model saved as: {model_filename}")

# Log metrics
os.makedirs('metrics', exist_ok=True)
metrics_file = "metrics/history.csv"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

new_metrics = pd.DataFrame({
    'timestamp': [timestamp],
    'pr_auc': [auc(recall, precision)]
})

if os.path.exists(metrics_file):
    existing_metrics = pd.read_csv(metrics_file)
    combined_metrics = pd.concat([existing_metrics, new_metrics], ignore_index=True)
else:
    combined_metrics = new_metrics

combined_metrics.to_csv(metrics_file, index=False)
print(f"Metrics logged to: {metrics_file}")