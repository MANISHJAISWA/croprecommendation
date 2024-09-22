import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load dataset and prepare model
data = pd.read_csv('crop_data.csv')  # Replace with your dataset filename
print(data.columns)  # Print the columns to check for 'Crop'

# If necessary, you can rename the columns here
# data.columns = [col.strip() for col in data.columns]  # Stripping spaces

# Check if 'Crop' exists, then proceed
if 'Crop' not in data.columns:
    raise ValueError("The dataset must contain a column named 'Crop'.")

data.fillna(method='ffill', inplace=True)
X = data.drop('Crop', axis=1)
y = data['Crop']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    input_data = [[data['temperature'], data['rainfall'], data['ph'], data['n'], data['p'], data['k']]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return jsonify({'recommended_crop': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

