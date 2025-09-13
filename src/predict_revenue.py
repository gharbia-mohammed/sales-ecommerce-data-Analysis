#E-commerce Revenue Prediction Project

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

#STEP 1: Aggregate and Prepare Monthly Data:

from google.colab import files
uploaded = files.upload()
df = pd.read_csv("sales_ecommerce_data.csv")
df = pd.read_csv("sales_ecommerce_data.csv", parse_dates=['Transaction_Date'])



#Aggregate data by month:

df['Month'] = df['Transaction_Date'].dt.to_period('M')
monthly_df = df.groupby('Month').agg({
    'Units_Sold': 'sum',
    'Ad_Spend': 'sum',
    'Conversion_Rate': 'mean',
    'Revenue': 'sum'
}).reset_index()

# Convert month back to timestamp for modeling:
monthly_df['Month'] = monthly_df['Month'].dt.to_timestamp()
print(monthly_df)



# STEP 2: Train and Evaluate Linear Regression Model

# Define predictors (X) and target (y) :

X = monthly_df[['Units_Sold', 'Ad_Spend', 'Conversion_Rate']]
y = monthly_df['Revenue']

# Train/Test split (use last month for testing):

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/12, shuffle=False
)

# Train the model:
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))



# STEP 3: Predict Revenue Growth with 30% Higher Ad Spend:

# Simulate scenario: +30% ad spend:
X_scenario = X.copy()
X_scenario['Ad_Spend'] *= 1.3
y_scenario_pred = model.predict(X_scenario)

print("Total Revenue 2024 (Actual):", total_2024)
print("Total Revenue 2024 (With +30% Ad Spend):", y_scenario_pred.sum())
print("Expected Growth from Ad Spend Increase:",
      (y_scenario_pred.sum() - total_2024) / total_2024 * 100)




# STEP 4: Visualize 2024 vs. Predicted 2025 Revenue (+30% Ad Spend):

X_2025 = X.copy()
X_2025['Ad_Spend'] *= 1.3  # +30% ad spend
y_2025_pred_30 = model.predict(X_2025)

# Build comparison DataFrame :
comparison_df = pd.DataFrame({
    'Month_2024': monthly_df['Month'],
    'Revenue_2024': y,
    'Predicted_Revenue_2025_AdSpend30': y_2025_pred_30
})

# Plot line chart comparison
plt.figure(figsize=(10, 5))

# Plot 2024 revenue
plt.plot(comparison_df['Month_2024'], comparison_df['Revenue_2024'],
         marker='o', label='Actual Revenue 2024', color='blue')

# Plot 2025 revenue (+30% ad spend)
plt.plot(comparison_df['Month_2024'], comparison_df['Predicted_Revenue_2025_AdSpend30'],
         marker='o', label='Predicted Revenue 2025', color='green')

# Title & labels
plt.title("Monthly Revenue: 2024 vs. Predicted 2025 (with 30% Increase in Ad Spend)")
plt.xlabel("Month")
plt.ylabel("Monthly Revenue ($)")

# Format y-axis
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Add more evenly spaced y-axis divisions
y_min = min(comparison_df['Revenue_2024'].min(), comparison_df['Predicted_Revenue_2025_AdSpend30'].min())
y_max = max(comparison_df['Revenue_2024'].max(), comparison_df['Predicted_Revenue_2025_AdSpend30'].max())
plt.yticks(np.linspace(y_min, y_max, 10))

# Rotate x-axis labels for readability
plt.xticks(rotation=0)

# Grid, legend, layout
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

