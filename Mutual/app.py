

from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load your trained model
import pickle

# Load the preprocessor from the pickle file
with open('preprocessor.pkl', 'rb') as file:
    loaded_encoder = pickle.load(file)

# Load label encoder for categorical columns
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# Assuming you have a training dataset, you need to fit the label encoder on the training data
# For example, assuming you have a DataFrame called 'train_data'
# label_encoder.fit(train_data[['occupation', 'gender']])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        income = float(request.form['income'])
        occupation = request.form['occupation']
        gender = request.form['gender']
        family_size = int(request.form['family_size'])
        risk_appetite = int(request.form['risk_appetite'])

        # Encode categorical variables
        # Fit the label encoder on the training data and then transform the new data
        

        # Make prediction
        input_data = pd.DataFrame({
            'age': [age],
            'income': [income],
            'occupation': [occupation],
            'gender': [gender],
            'family_size': [family_size],
            'risk_appetite': [risk_appetite]
        })
        input_data = pd.DataFrame(loaded_encoder.transform(input_data),columns=loaded_encoder.get_feature_names_out())
        print (input_data)



        prediction = loaded_model.predict(input_data)
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
