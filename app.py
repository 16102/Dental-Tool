import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

app = Flask(__name__)

# Load models
model_diagnosis = pickle.load(open('model.pkl', 'rb'))
model_treatment = pickle.load(open('model-treatment.pkl', 'rb'))

# Load dataset for scaling and label encoding
dataset = pd.read_csv('dental-data.csv')  # Ensure this dataset matches your feature needs

# Label encoding for categorical columns
label_encoders = {}
for col in [1, 2, 3, 4, 5, 6, 7]:
    le = LabelEncoder()
    dataset.iloc[:, col] = le.fit_transform(dataset.iloc[:, col])
    label_encoders[col] = le

# Scaling for numerical columns
numerical_columns = [0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
sc = MinMaxScaler(feature_range=(0, 1))
sc.fit(dataset.iloc[:, numerical_columns])

# Label encoding for diagnosis and treatment columns
label_encoder_diagnosis = LabelEncoder()
dataset['Diagnosis'] = label_encoder_diagnosis.fit_transform(dataset.iloc[:, 24])
label_encoder_treatment = LabelEncoder()
dataset['Treatment'] = label_encoder_treatment.fit_transform(dataset.iloc[:, 23])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/treatment')
def treatment():
    return render_template('treatment.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        string_features = [request.form.get(f'input_{i}') for i in [1, 2, 3, 4, 5, 6, 7]]
        float_features = [request.form.get(f'input_{i}') for i in list(range(0, 1)) + list(range(8, 23))]

        # Convert float features to numeric and handle missing values
        float_features = [float(f) if f else 0.0 for f in float_features]

        # Convert string features using label encoders and handle missing values
        encoded_features = [label_encoders[i].transform([f])[0] if f in label_encoders[i].classes_ else 0 for i, f in zip([1, 2, 3, 4, 5, 6, 7], string_features)]

        # Combine numerical and encoded categorical features
        combined_features = float_features + encoded_features

        # Scale numerical features
        scaled_numerical_features = sc.transform([combined_features[:len(numerical_columns)]])
        
        # Combine scaled numerical features and encoded categorical features
        final_features = np.concatenate((scaled_numerical_features[0], combined_features[len(numerical_columns):])).reshape(1, -1)

        # Predictions
        diagnosis_prediction = model_diagnosis.predict(final_features)

        # Decode predictions
        diagnosis_output = label_encoder_diagnosis.inverse_transform(diagnosis_prediction)[0]

        return render_template('diagnosis.html', prediction_text=f'Predicted Diagnosis : {diagnosis_output}')
    
    except Exception as e:
        return render_template('diagnosis.html', prediction_text=f'Error: {e}')

@app.route('/submit_treatment', methods=['POST'])
def submit_treatment():
    try:
        # Extract features from form
        string_features = [request.form.get(f'input_{i}_treatment') for i in [1, 2, 3, 4, 5, 6, 7]]
        float_features = [request.form.get(f'input_{i}_treatment') for i in list(range(0, 1)) + list(range(8, 23))]

        diagnosis_feature = request.form.get('input_23_treatment')
        # Convert float features to numeric and handle missing values
        float_features = [float(f) if f else 0.0 for f in float_features]

        # Convert string features using label encoders and handle missing values
        encoded_features = [label_encoders[i].transform([f])[0] if f in label_encoders[i].classes_ else 0 for i, f in zip([1, 2, 3, 4, 5, 6, 7], string_features)]

        encoded_diagnosis = label_encoder_diagnosis.transform([diagnosis_feature])[0] if diagnosis_feature in label_encoder_diagnosis.classes_ else 0
        # Combine numerical and encoded categorical features
        combined_features = float_features + encoded_features + [encoded_diagnosis]

        # Scale numerical features
        scaled_numerical_features = sc.transform([combined_features[:len(numerical_columns)]])
        
        # Combine scaled numerical features and encoded categorical features
        final_features = np.concatenate((scaled_numerical_features[0], combined_features[len(numerical_columns):])).reshape(1, -1)

        # Predictions
        treatment_prediction = model_treatment.predict(final_features)

        # Decode predictions
        treatment_output = label_encoder_treatment.inverse_transform(treatment_prediction)[0]

        return render_template('treatment.html', treatment_text=f'Predicted Treatment Plan: {treatment_output}')
    
    except Exception as e:
        return render_template('treatment.html', treatment_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
