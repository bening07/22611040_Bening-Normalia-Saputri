import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/Users/user/Documents/UAS MPML/weather_classification_data.csv')

st.title('Weather Classification Analysis')

# Display basic information
st.write("Dataset Info:")
st.write(df.info())
st.write(df.describe())
st.write(df.head())

# Plot distributions of numerical features
st.write("Distribution of Numerical Features:")
numerical_features = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
fig, ax = plt.subplots(len(numerical_features), 1, figsize=(10, 20))
for i, col in enumerate(numerical_features):
    sns.histplot(df[col], ax=ax[i])
st.pyplot(fig)

# Plot categorical features
st.write("Distribution of Categorical Features:")
categorical_features = ['Cloud Cover', 'Season', 'Location', 'Weather Type']
for col in categorical_features:
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=col, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Correlation heatmap for numerical features
st.write("Correlation Heatmap of Numerical Features:")
correlation_matrix = df[numerical_features].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Pairplot of numerical features colored by Weather Type
st.write("Pairplot of Numerical Features Colored by Weather Type:")
sns.pairplot(df, hue='Weather Type', vars=numerical_features)
st.pyplot()
