from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html', options=encoders)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = []

    for col in columns:
        val = data.get(col)
        if col in encoders:
            val = encoders[col].transform([val])[0]
        else:
            val = float(val)
        input_data.append(val)

    prediction = model.predict([input_data])[0]
    return render_template('index.html', prediction=round(prediction,2), options=encoders)

if __name__ == "__main__":
    app.run(debug=True)
