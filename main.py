import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("train.csv")

# STEP 1: Choose inputs (features)
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# STEP 2: Output (what we predict)
y = data['SalePrice']

# STEP 3: Split data (training + testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# STEP 4: Create model
model = LinearRegression()

# STEP 5: Train model
model.fit(X_train, y_train)

# STEP 6: Test prediction
predictions = model.predict(X_test)

print("Model trained successfully!")
print(predictions[:5])

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

sample_house = pd.DataFrame([[2000, 3, 2]],
columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])
price = model.predict(sample_house)

print("Predicted price for sample house:", price[0])