

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


heart_df = pd.read_csv('heart.csv')

# Preprocessing and splitting the dataset
x = heart_df.drop('target', axis=1)
y = heart_df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

# Decision Tree with Hyperparameter Tuning
dtc = DecisionTreeClassifier()
grid_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(2, 32, 1),
    'min_samples_leaf': range(1, 10, 1),
    'min_samples_split': range(2, 10, 1),
    'splitter': ['best', 'random']
}
grid_search = GridSearchCV(dtc, grid_params, cv=10, n_jobs=-1, verbose=1)
grid_search.fit(x_train, y_train)

best_dtc = grid_search.best_estimator_
y_pred_dtc = best_dtc.predict(x_test)

accuracy_dtc = accuracy_score(y_test, y_pred_dtc)
print("Decision Tree Accuracy:", accuracy_dtc)
print("Classification Report (Decision Tree):\n", classification_report(y_test, y_pred_dtc))

# Random Forest
rfc = RandomForestClassifier(criterion='gini', max_depth=7, max_features='sqrt', min_samples_leaf=2, min_samples_split=4, n_estimators=180)
rfc.fit(x_train, y_train)

y_pred_rfc = rfc.predict(x_test)

accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
print("Random Forest Accuracy:", accuracy_rfc)
print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rfc))




# Load the dataset
heart_data = pd.read_csv('heart.csv')

# Preprocessing and splitting the dataset
features = heart_data.drop('target', axis=1)
target = heart_data['target']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=0)

# Decision Tree with Hyperparameter Tuning
dt_classifier = DecisionTreeClassifier()
dt_grid_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(2, 32, 1),
    'min_samples_leaf': range(1, 10, 1),
    'min_samples_split': range(2, 10, 1),
    'splitter': ['best', 'random']
}
dt_grid_search = GridSearchCV(dt_classifier, dt_grid_params, cv=10, n_jobs=-1, verbose=1)
dt_grid_search.fit(X_train, y_train)

best_dt_classifier = dt_grid_search.best_estimator_
y_pred_dt = best_dt_classifier.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)
print("Classification Report (Decision Tree):\n", classification_report(y_test, y_pred_dt))

# Random Forest
rf_classifier = RandomForestClassifier(criterion='gini', max_depth=7, max_features='sqrt', min_samples_leaf=2, min_samples_split=4, n_estimators=180)
rf_classifier.fit(X_train, y_train)

y_pred_rf = rf_classifier.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))
