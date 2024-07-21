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
from sklearn.model_selection import GridSearchCV

st.set_page_config(
    page_title="Weather Classification",
    page_icon="☁️",
    layout="wide"
)

st.title('☁️ Weather Classification Data Exploration, Preprocessing, and Model Evaluation')
st.write("Welcome to the weather classification app. This application allows you to explore, preprocess, and evaluate various machine learning models on weather data.")

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv('weather_classification_data.csv')
    return df

df = load_data()

# Display basic information
st.header('Data Overview')
st.write("Here is the basic information about the dataset:")
st.write(df.head())

with st.expander("View detailed information"):
    st.write(df.info())
    st.write(df.describe())

# Plot distributions of numerical features
st.header('Distribution of Numerical Features')
numerical_features = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
fig, axes = plt.subplots(len(numerical_features), 1, figsize=(10, 10))

for i, col in enumerate(numerical_features):
    df[col].hist(bins=30, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

st.pyplot(fig)

# Plot categorical features
st.header('Distribution of Categorical Features')
categorical_features = ['Cloud Cover', 'Season', 'Location', 'Weather Type']
for col in categorical_features:
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=col, ax=ax)
    ax.set_title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Correlation heatmap for numerical features
st.header('Correlation Heatmap of Numerical Features')
correlation_matrix = df[numerical_features].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap of Numerical Features')
st.pyplot(fig)

# Preprocessing section
st.header('Preprocessing')
st.write("In this section, we preprocess the data by handling missing values and scaling features.")

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
st.write(f'Processed data shape: {X_processed.shape}')

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
st.header('Model Performance')
st.write("Here are the performance metrics for each model:")

for model_name, metrics in results.items():
    st.subheader(model_name)
    st.write(f"**Accuracy:** {metrics['Accuracy']:.4f}")
    st.write(f"**Precision:** {metrics['Precision']:.4f}")
    st.write(f"**Recall:** {metrics['Recall']:.4f}")
    st.write(f"**F1 Score:** {metrics['F1 Score']:.4f}")

# Plot performance
results_df = pd.DataFrame(results).T
st.header('Model Performance Comparison')
st.bar_chart(results_df)

# Hyperparameter tuning section
st.header('Hyperparameter Tuning')
st.write("In this section, we perform hyperparameter tuning to find the best model configurations.")

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
st.header('Tuned Model Performance')
st.write("Here are the performance metrics for the best models after hyperparameter tuning:")

for model_name, metrics in results.items():
    st.subheader(model_name)
    st.write(f"**Accuracy:** {metrics['Accuracy']:.4f}")
    st.write(f"**Precision:** {metrics['Precision']:.4f}")
    st.write(f"**Recall:** {metrics['Recall']:.4f}")
    st.write(f"**F1 Score:** {metrics['F1 Score']:.4f}")
