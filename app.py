import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

st.set_page_config(page_title='Weather Classification', layout='wide')
st.title('🌤️ Weather Classification Data Exploration, Preprocessing, and Model Evaluation')

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv('weather_classification_data.csv')
    return df

df = load_data()

# Display basic information
st.sidebar.header('Dataset Information')
buffer = []
df.info(buf=buffer)
s = buffer.getvalue()
st.sidebar.text(s)
st.sidebar.write(df.describe())
st.sidebar.write(df.head())

# Plot distributions of numerical features
st.subheader('Distribution of Numerical Features')
numerical_features = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
for col in numerical_features:
    fig, ax = plt.subplots()
    df[col].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
    ax.set_title(f'Distribution of {col}', fontsize=14)
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    st.pyplot(fig)

# Plot categorical features
st.subheader('Distribution of Categorical Features')
categorical_features = ['Cloud Cover', 'Season', 'Location', 'Weather Type']
for col in categorical_features:
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=col, ax=ax, palette='viridis')
    ax.set_title(f'Distribution of {col}', fontsize=14)
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Correlation heatmap for numerical features
st.subheader('Correlation Heatmap of Numerical Features')
correlation_matrix = df[numerical_features].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap of Numerical Features', fontsize=16)
st.pyplot(fig)

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

# Train and evaluate models with cross-validation
st.subheader('Model Performance with Cross-Validation')
for model_name, model in models.items():
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

# Display results
for model_name, metrics in results.items():
    st.write(f"### {model_name}")
    for metric_name, value in metrics.items():
        st.write(f"{metric_name}: {value:.4f}")
    st.write("\n")

# Plot performance using matplotlib for better control
results_df = pd.DataFrame(results).T

# Transpose the DataFrame for vertical bar chart
results_df = results_df.T

st.subheader('Model Performance Comparison')

fig, ax = plt.subplots(figsize=(10, 6))
results_df.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], edgecolor='black')
ax.set_title('Model Performance Comparison', fontsize=16)
ax.set_xlabel('Metric', fontsize=14)
ax.set_ylabel('Score', fontsize=14)
plt.xticks(rotation=0)
st.pyplot(fig)

# Hyperparameter tuning section
st.subheader('Hyperparameter Tuning')

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

# Evaluate tuned models with cross-validation
st.subheader('Tuned Model Performance with Cross-Validation')
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
    st.write(f"### {model_name}")
    for metric_name, value in metrics.items():
        st.write(f"{metric_name}: {value:.4f}")
    st.write("\n")

# Final thoughts and conclusions
st.subheader('Conclusions')
st.write("""
    This application allows for the exploration, preprocessing, and evaluation of different machine learning models on weather classification data.
    The visualizations and performance metrics provide insights into the effectiveness of various models and preprocessing techniques.
    Cross-validation ensures the robustness of the model evaluations.
""")
