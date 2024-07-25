import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv('/Users/user/Documents/UAS MPML/weather_classification_data.csv')

# Mengecek apakah ada nilai yang hilang di setiap kolom
missing_values = df.isnull().sum()

# Menampilkan kolom yang memiliki nilai hilang dan jumlahnya
print("Missing values in each column:")
print(missing_values[missing_values > 0])

# Menampilkan jumlah total nilai yang hilang dalam dataset
total_missing = missing_values.sum()
print(f"\nTotal missing values in the dataset: {total_missing}")

# Identify features
numerical_features = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
                      'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
categorical_features = ['Cloud Cover', 'Season', 'Location']

# Define preprocessing for numerical and categorical features
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

X = df.drop(columns=['Weather Type'])
y = df['Weather Type']

X_processed = preprocessor.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

results = {}

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# Display results
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("\n")

import matplotlib.pyplot as plt

# Convert results to DataFrame
results_df = pd.DataFrame(results).T

# Plot performance
results_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.show()

from sklearn.model_selection import GridSearchCV

# Define parameter grids for tuning
param_grids = {
    "Decision Tree": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance']
    }
}

# Hyperparameter tuning
best_models = {}
for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_

# Evaluate tuned models with cross-validation
for model_name, model in best_models.items():
    accuracies = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    precisions = cross_val_score(model, X_train, y_train, cv=5, scoring='precision_weighted')
    recalls = cross_val_score(model, X_train, y_train, cv=5, scoring='recall_weighted')
    f1s = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')

    results[model_name] = {
        "Accuracy": accuracies.mean(),
        "Precision": precisions.mean(),
        "Recall": recalls.mean(),
        "F1 Score": f1s.mean()
    }

# Display results of tuned models
for model_name, metrics in results.items():
    print(f"Model: {model_name} (Tuned)")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("\n")
