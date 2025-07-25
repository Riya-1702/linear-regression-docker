import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os # For path manipulation


# Load the data
try:
    df = pd.read_csv('advertising_sales_data.csv')
except FileNotFoundError:
    print("Error: 'advertising_sales_data.csv' not found. Please make sure it's in the same directory.")
    print("You can copy the CSV data provided previously and save it as 'advertising_sales_data.csv'.")
    exit() # Exit if data file is missing

# Define features (X) and target (y)
X = df[['TV_Ad_Spend', 'Radio_Ad_Spend', 'Newspaper_Ad_Spend']]
y = df['Sales_Units']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
print("\nTraining the Linear Regression Model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the model's coefficients and intercept
print("\n--- Model Coefficients ---")
print("Intercept:", model.intercept_)
print("Coefficients (TV, Radio, Newspaper):", model.coef_)



plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Sales Units")
plt.ylabel("Predicted Sales Units")
plt.title("Actual vs. Predicted Sales Units (Advertising Data)")
plt.grid(True)

# Define output directory and file path
output_dir = 'output_plots'
os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist
plot_file_path = os.path.join(output_dir, 'advertising_sales_prediction_plot.png')
plt.savefig(plot_file_path)
print(f"\nPlot saved to {plot_file_path}")

# plt.show() # Uncomment if you have X11 forwarding set up or are in an interactive environment



print("\n--- Predict Sales for New Advertising Spend ---")
try:
    tv_spend = float(input("Enter TV Ad Spend (e.g., 200): "))
    radio_spend = float(input("Enter Radio Ad Spend (e.g., 30): "))
    newspaper_spend = float(input("Enter Newspaper Ad Spend (e.g., 50): "))

    # Create a DataFrame for the new input
    new_data = pd.DataFrame([[tv_spend, radio_spend, newspaper_spend]],
                            columns=['TV_Ad_Spend', 'Radio_Ad_Spend', 'Newspaper_Ad_Spend'])

    # Make prediction using the trained model
    predicted_sales = model.predict(new_data)
    print(f"\nPredicted Sales Units for the given spend: {predicted_sales[0]:.2f}")

except ValueError:
    print("Invalid input. Please enter numeric values for ad spend.")

except Exception as e:
    print(f"An unexpected error occurred during prediction: {e}")

print("\nScript execution finished.")
