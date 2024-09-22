import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv('crop_data.csv')
data.fillna(method='ffill', inplace=True)
data = pd.get_dummies(data)

# Split features and labels
X = data.drop('Crop', axis=1)
y = data['Crop']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Function for crop recommendation
def recommend_crop(temp, rainfall, ph, n, p, k):
    input_data = [[temp, rainfall, ph, n, p, k]]  # Adjust based on your feature set
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]

# Example usage
recommended_crop = recommend_crop(25, 150, 6.5, 50, 30, 20)
print(f"Recommended crop: {recommended_crop}")
