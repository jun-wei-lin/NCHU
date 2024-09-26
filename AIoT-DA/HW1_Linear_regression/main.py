##Step 1: Import Libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Title of the app
st.title("Linear Regression Demo")

# Input parameters
st.sidebar.header("Input Parameters")
slope = st.sidebar.slider("Slope", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
intercept = st.sidebar.slider("Intercept", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
noise_level = st.sidebar.slider("Noise Level", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
y = intercept + slope * X + noise_level * np.random.randn(100, 1)  # Linear relation with noise

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Convert data to DataFrame for plotting
data = pd.DataFrame(np.column_stack((X, y)), columns=['X', 'y'])

# Display the scatter plot of the actual data
st.subheader("Scatter Plot of Data")
st.scatter_chart(data)

# Plotting the regression line
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Actual data')
ax.plot(X_test, y_pred, color='red', label='Predicted line', linewidth=2)
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Linear Regression Prediction')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# Display coefficients
st.write(f"Coefficient: {model.coef_[0][0]}")
st.write(f"Intercept: {model.intercept_[0]}")

##Step 2: Generate Synthetic Data

# Set a random seed for reproducibility
np.random.seed(42)

# Generate some synthetic data
X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship with some noise

#Step 3: Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 4: Create and Fit the Linear Regression Model
# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

#Step 5: Make Predictions
# Predict using the test data
y_pred = model.predict(X_test)


#Step 6: Visualize the Results
# Plotting the results
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', label='Predicted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Prediction')
plt.legend()
plt.show()

#Step 7: Evaluate the Model
# Calculate the coefficients and intercept
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

# Calculate the R^2 score
r_squared = model.score(X_test, y_test)
print("R^2 Score:", r_squared)
