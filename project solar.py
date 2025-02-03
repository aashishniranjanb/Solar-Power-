# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Make sure to replace 'solar_power_data.csv' with your actual dataset file
data = pd.read_csv("C:/Users/Home/Desktop/New folder/03f4d1c1a55947025601.csv")

# Display the first few rows of the dataset
print("Initial Data:")
print(data.head())

# **Data Cleaning Steps**

# Step 1: Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Step 2: Handle missing values
# Option 1: Fill missing values with the mean (for numerical columns)
data.fillna(data.mean(), inplace=True)

# Option 2: Drop rows with missing values (uncomment if preferred)
# data.dropna(inplace=True)

# Step 3: Remove duplicates
data.drop_duplicates(inplace=True)

# Step 4: Convert data types if necessary
# Example: Convert 'Date-Time' column to datetime format
if 'Date-Time' in data.columns:
    data['Date-Time'] = pd.to_datetime(data['Date-Time'])

# Step 5: Remove outliers (optional)
# Define a function to identify outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Example: Remove outliers from 'generated_power_kw' column
if 'generated_power_kw' in data.columns:
    data = remove_outliers(data, 'generated_power_kw')

# Step 6: Reset index after cleaning
data.reset_index(drop=True, inplace=True)

# Display the cleaned dataset
print("\nCleaned Data:")
print(data.head())

# Data Visualization Steps

# Set the style for seaborn
sns.set(style="whitegrid")

# 1. Time Series Plot of Solar Power Generation
plt.figure(figsize=(14, 6))
plt.plot(data['Date-Time'], data['generated_power_kw'], color='orange', label='Generated Power (kW)')
plt.title('Solar Power Generation Over Time')
plt.xlabel('Date-Time')
plt.ylabel('Power Output (kW)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Scatter Plot of Solar Power vs. Solar Irradiance
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GHI', y='generated_power_kw', data=data, color='blue', alpha=0.6)
plt.title('Solar Power Generation vs. Global Horizontal Irradiance')
plt.xlabel('Global Horizontal Irradiance (W/m²)')
plt.ylabel('Generated Power (kW)')
plt.grid()
plt.show()

# 3. Box Plot of Solar Power Generation by Month
data['Month'] = pd.to_datetime(data['Date-Time']).dt.month
plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='generated_power_kw', data=data, palette='Set2')
plt.title('Solar Power Generation Distribution by Month')
plt.xlabel('Month')
plt.ylabel('Generated Power (kW)')
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid()
plt.show()

# 4. Heatmap of Correlation Matrix
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap of Solar Power Dataset Features')
plt.show()

#Model Linear Regression
# Select relevant features for prediction
features = [
    'temperature_2_m_above_gnd',
    'relative_humidity_2_m_above_gnd',
    'mean_sea_level_pressure_MSL',
    'total_precipitation_sfc',
    'snowfall_amount_sfc',
    'total_cloud_cover_sfc',
    'shortwave_radiation_backwards_sfc',
    'wind_speed_10_m_above_gnd'
]

# Define the target variable
target = 'generated_power_kw'

# Prepare the feature matrix (X) and target vector (y)
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training sets
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Output the coefficients
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)

# Calculate and print the mean squared error and R-squared value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean squared error: %.2f" % mse)
print("R-squared: %.2f" % r2)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='black', label='Actual vs Predicted')
plt.plot(y_test, y_test, color='blue', linewidth=3, label='Perfect Prediction')
plt.title('Solar Power Prediction using Linear Regression')
plt.xlabel('Actual Power Output (kW)')
plt.ylabel('Predicted Power Output (kW)')
plt.legend()
plt.grid()
plt.show()