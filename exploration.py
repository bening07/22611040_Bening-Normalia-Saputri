import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/Users/user/Documents/UAS MPML/weather_classification_data.csv')

# Display basic information
print(df.info())
print(df.describe())
print(df.head())

# Plot distributions of numerical features
numerical_features = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
df[numerical_features].hist(bins=30, figsize=(15, 10))
plt.suptitle('Distribution of Numerical Features')
plt.show()

# Plot categorical features
categorical_features = ['Cloud Cover', 'Season', 'Location', 'Weather Type']
for col in categorical_features:
    sns.countplot(data=df, x=col)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()

# Correlation heatmap for numerical features
correlation_matrix = df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Pairplot of numerical features colored by Weather Type
sns.pairplot(df, hue='Weather Type', vars=numerical_features)
plt.title('Pairplot of Numerical Features Colored by Weather Type')
plt.show()
