
E-Commerce Data Cleaning & Feature Engineering

import pandas as pd

# 1️ Load Dataset

df = pd.read_csv("sales_ecommerce_data.csv")

# 2️⃣ Explore Dataset

print("📊 Dataset Shape:", df.shape)
print("\nℹ️ Dataset Info:")
print(df.info())
print("\n❓ Missing Values per Column:\n", df.isnull().sum())
print("\n🔁 Number of Duplicate Rows:", df.duplicated().sum())


# 3️⃣ Data Cleaning

# Remove duplicate rows :
df = df.drop_duplicates()

# Fill missing values
df = df.fillna({
    "price": df["price"].mean(),            # Replace missing prices with mean
    "category": df["category"].mode()[0]    # Replace missing categories with most frequent
})

# Convert order_date to datetime
df["order_date"] = pd.to_datetime(df["order_date"])

# 4️⃣ Feature Engineering

# Extract date components

df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.month
df["day"] = df["order_date"].dt.day

# Create new metrics
df["revenue"] = df["quantity"] * df["price"]
df["net_revenue"] = df["revenue"] * (1 - df["discount"])
df["cpc"] = df["ad_spend"] / df["clicks"]                 # Cost per Click
df["ctr"] = df["clicks"] / df["impressions"]              # Click Through Rate
df["conversion_rate"] = df["conversions"] / df["clicks"]  # Conversion Rate

# 5️⃣ Save Clean Dataset
# ------------------------------
df.to_csv("cleaned_ecommerce.csv", index=False)
print("✅ Clean dataset saved as 'cleaned_ecommerce.csv'")
