import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. LOAD DATA
# ==========================================
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ==========================================
# 2. FEATURE SELECTION & ENGINEERING
# ==========================================
# We include the 'Task 01' features plus the most influential ones from the dataset
features = [
    'GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 
    'OverallQual', 'YearBuilt', 'GarageCars', 'TotalBsmtSF'
]

def clean_data(df):
    X = df[features].copy()
    
    # 1. Combine Baths
    X['TotalBath'] = X['FullBath'] + (0.5 * X['HalfBath'])
    X.drop(['FullBath', 'HalfBath'], axis=1, inplace=True)
    
    # 2. Handle missing values 
    # (e.g., GarageCars might be NaN in test set if house has no garage)
    X = X.fillna(X.median())
    
    return X

X = clean_data(train)
X_test_final = clean_data(test)
y = np.log1p(train['SalePrice'])

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
# Let's visualize how the most important feature (Quality) relates to Price
plt.figure(figsize=(10, 6))
sns.boxplot(x=train['OverallQual'], y=train['SalePrice'])
plt.title('House Quality vs Sale Price')
plt.show()



# ==========================================
# 4. TRAINING
# ==========================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# 5. VALIDATION
# ==========================================
# Calculate RMSE on log scale
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))

print(f"--- Model Performance ---")
print(f"Validation RMSE: {rmse:.4f}")

# Show which features the model values most
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nFeature Importance:")
print(coeff_df.sort_values(by='Coefficient', ascending=False))

# ==========================================
# 6. GENERATE SUBMISSION
# ==========================================
final_preds = np.expm1(model.predict(X_test_final))

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': final_preds
})

submission.to_csv('submission_v2.csv', index=False)
print("\nAdvanced submission file 'submission_v2.csv' is ready!")