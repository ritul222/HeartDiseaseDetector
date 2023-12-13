
# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

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

# Random Forest
rf_classifier = RandomForestClassifier(criterion='gini', max_depth=7, max_features='sqrt', min_samples_leaf=2, min_samples_split=4, n_estimators=180)
rf_classifier.fit(X_train, y_train)

# Load the pre-trained models
dt_model = best_dt_classifier
rf_model = rf_classifier

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]

        # Make predictions using both models
        prediction_dt = dt_model.predict([features])[0]
        prediction_rf = rf_model.predict([features])[0]

        return render_template('result.html', prediction_dt=prediction_dt, prediction_rf=prediction_rf)

if __name__ == '__main__':
    app.run(debug=True)







