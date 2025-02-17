# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style="whitegrid")

# Assuming your dataset is in a CSV file called 'call_of_duty.csv'
df = pd.read_csv('cod.csv', index_col=0)
df.head()

# Plot histograms for numerical features
for feature in df.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature].dropna(), bins=30, kde=True, color='skyblue')
    plt.title(f'Distribution of {feature.capitalize()}')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Frequency')
    plt.show()

# Log transformation due to skewed distribution
# df = df.apply(np.log1p)

corr_matrix = df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix,
            annot=True,        # Display correlation values in the heatmap
            cmap='coolwarm',   # Color map
            fmt=".2f",         # Format the annotation to 2 decimal places
            linewidths=.5)

plt.title('Pairwise Correlation Heatmap')
plt.show()

# scaler = StandardScaler()

# df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Scale the X variables

scaler = StandardScaler()

# Separate features (X) and target variable (y)
X_var = df.drop(columns=['level'])  # Features to scale
y_var = df['level']  # Target variable (unchanged)

# Scale only the feature columns
X_scaled = pd.DataFrame(scaler.fit_transform(X_var), columns=X_var.columns, index=X_var.index)

# Add the unscaled target variable back to the DataFrame
df_scaled = X_scaled.copy()
df_scaled['level'] = y_var

df_scaled.head()

# Ridge Regression
X = df_scaled.drop(columns=['level'])  # Features
y = df_scaled['level']  # Target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Ridge Regression with alpha (regularization strength)
ridge = Ridge(alpha=1.0)  # You can tune alpha

# Train the model
ridge.fit(X_train, y_train)

# Predict on test data
y_pred = ridge.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)

print(f'R-squared Score: {r2}')

# Create a DataFrame with feature names and their Ridge coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": ridge.coef_
})

# Sort by absolute coefficient value
coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=False).index)

print(coef_df)

# Sort features correctly from smallest to largest coefficient
coef_df = coef_df.sort_values(by="Coefficient", ascending=False)

# Set up the figure size
plt.figure(figsize=(8, 4.5))

# Create a horizontal barplot with Seaborn
sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="coolwarm", orient="h")

# Customize labels and title
plt.title("Feature Coefficients from Ridge Model for Player Level", fontsize=14)
plt.xlabel("Coefficient", fontsize=12)
plt.ylabel("Feature", fontsize=12)

# Add grid for better readability
plt.grid(axis='x', which='major', linestyle='--', alpha=0.75)

# Adjust layout for a cleaner look
plt.tight_layout()

# Show the plot
plt.show()

# Removing XP due to knowing that more XP is directly related to going up in levels
# Ridge regression without XP
X2 = df_scaled.drop(columns=['level', 'xp'])  # Features
y2 = df_scaled['level']  # Target

# Split data into train and test sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Initialize Ridge Regression with alpha (regularization strength)
ridge2 = Ridge(alpha=1.0)

# Train the model
ridge2.fit(X_train2, y_train2)

# Predict on test data
y_pred2 = ridge2.predict(X_test2)

# Evaluate the model
r2_2 = r2_score(y_test2, y_pred2)

print(f'R-squared Score: {r2_2}')

# Create a DataFrame with feature names and their Ridge coefficients
coef_df2 = pd.DataFrame({
    "Feature": X2.columns,
    "Coefficient": ridge2.coef_
})

# Sort by absolute coefficient value
coef_df2 = coef_df2.reindex(coef_df2["Coefficient"].abs().sort_values(ascending=False).index)

print(coef_df2)

# Sort features correctly from smallest to largest coefficient
coef_df2 = coef_df2.sort_values(by="Coefficient", ascending=False)

# Set up the figure size
plt.figure(figsize=(8, 4.5))

# Create a horizontal barplot with Seaborn
sns.barplot(x="Coefficient", y="Feature", data=coef_df2, palette="coolwarm", orient="h")

# Customize labels and title
plt.title("Feature Coefficients from Ridge Model for Player Level", fontsize=14)
plt.xlabel("Coefficient", fontsize=12)
plt.ylabel("Feature", fontsize=12)

# Add grid for better readability
plt.grid(axis='x', which='major', linestyle='--', alpha=0.75)

# Adjust layout for a cleaner look
plt.tight_layout()

# Show the plot
plt.show()

