import pandas as pd
import numpy as np

print("Reading census data...")
df = pd.read_csv("data/census.csv")

# Remove spaces in col names
print("Removing extra spaces...")
df.columns = [c.strip() for c in df.columns]

# Remove spaces in categorical columns
for col in df.columns:
    if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
        df[col] = df[col].str.strip()

# Drop unknown/nans
print("Dropping Nans...")
df.replace("?", np.nan, inplace=True)
df.dropna(how="any", inplace=True)

# Save data to file
print("Saving data to /data/census_cleaned.csv")
df.to_csv("./data/census_cleaned.csv", index=False)
