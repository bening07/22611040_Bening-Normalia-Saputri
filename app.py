import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title('Weather Classification Data Exploration, Preprocessing, and Model Evaluation')

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv('weather_classification_data.csv')
    return df

df = load_data()

# Display basic information
st.write(df.info())
st.write(df.describe())
st.write(df.head())

# Plot distributions of numerical features
st.subheader('Distribution of Numerical Features')
numerical_features = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
for col in numerical_features:
    fig, ax = plt.subplots()
    df[col].hist(bins=30, ax=ax)
    ax.set_title(f'Distribution of {col}')
    st.pyplot(fig)

# Plot categorical features
st.subheader('Distribution of Categorical Features')
categorical_features = ['Cloud Cover', 'Season', 'Location', 'Weather Type']
for col in categorical_features:
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=col, ax=ax)
    ax.set_title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Correlation heatmap for numerical features
st.subheader('Correlation Heatmap of Numerical Features')
correlation_matrix = df[numerical_features].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap of Numerical Features')
st.pyplot(fig)

# Pairplot of numerical features colored by Weather Type
st.subheader('Pairplot of Numerical Features Colored by Weather Type')
sns.pairplot(df, hue='Weather Type', vars=numerical_features)
st.pyplot()

# Preprocessing section
st.subheader('Preprocessing')

# Identify features
numerical_features = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
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
st.write('Preprocessing done!')

st.write('Processed data shape:', X_processed.shape)

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
st.subheader('Model Performance')
for model_name, metrics in results.items():
    st.write(f"Model: {model_name}")
    for metric_name, value in metrics.items():
        st.write(f"{metric_name}: {value:.4f}")
    st.write("\n")

# Plot performance
results_df = pd.DataFrame(results).T
st.subheader('Model Performance Comparison')
st.bar_chart(results_df)

# Hyperparameter tuning section
st.subheader('Hyperparameter Tuning')
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

best_models = {}
for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_

# Evaluate tuned models
for model_name, model in best_models.items():
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

# Display results of tuned models
st.subheader('Tuned Model Performance')
for model_name, metrics in results.items():
    st.write(f"Model: {model_name}")
    for metric_name, value in metrics.items():
        st.write(f"{metric_name}: {value:.4f}")
    st.write("\n")
