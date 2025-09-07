import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_customers = 500
data = {
    'Campaign_Spend': np.random.uniform(10, 100, num_customers),
    'Engagement_Rate': np.random.uniform(0.1, 0.9, num_customers),
    'Purchase_Frequency': np.random.poisson(3, num_customers),
    'Average_Purchase_Value': np.random.uniform(20, 100, num_customers),
    'CLTV': np.random.uniform(100, 1000, num_customers) # Target variable
}
df = pd.DataFrame(data)
# Add some noise to the data to make it more realistic
df['CLTV'] += np.random.normal(0, 50, num_customers)
# --- 2. Data Cleaning and Feature Engineering ---
# No significant cleaning needed for synthetic data, but in real-world scenarios, 
# this step would involve handling missing values, outliers, etc.
# Feature Engineering: Calculate a simple CLTV proxy
df['CLTV_Proxy'] = df['Purchase_Frequency'] * df['Average_Purchase_Value']
# --- 3. Model Building ---
# Split data into training and testing sets
X = df[['Campaign_Spend', 'Engagement_Rate', 'Purchase_Frequency', 'Average_Purchase_Value']]
y = df['CLTV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a linear regression model (choose a suitable model based on data characteristics)
model = LinearRegression()
model.fit(X_train, y_train)
# --- 4. Model Evaluation ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
# --- 5. Visualization ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual CLTV")
plt.ylabel("Predicted CLTV")
plt.title("Actual vs. Predicted CLTV")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--') # Ideal line
plt.grid(True)
plt.tight_layout()
output_filename = 'cltv_prediction_plot.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Further analysis and model tuning would be performed in a real-world scenario.  This example provides a basic framework.