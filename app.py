# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'paymentdefault.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        LIMIT_BAL = int(request.form['LIMIT_BAL'])
        SEX = int(request.form['SEX'])
        EDUCATION = int(request.form['EDUCATION'])
        PAY_1 = int(request.form['PAY_1'])
        PAY_AMT1 = int(request.form['PAY_AMT1'])
        BILL_AMT1 = float(request.form['BILL_AMT1'])
        
        data = np.array([[LIMIT_BAL, SEX, EDUCATION, PAY_1, PAY_AMT1, BILL_AMT1]])
        my_prediction = classifier.predict(data)
        print(my_prediction)
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
