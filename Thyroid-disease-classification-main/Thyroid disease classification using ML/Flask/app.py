from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

model = pickle.load(open("thyroid_1_model.pkl", 'rb'))
le = pickle.load(open("label_encoder.pkl", 'rb'))

app = Flask(__name__)


@app.route('/')
def about():
    return render_template("Home.html")


@app.route("/Home.html")
def home():
    return render_template("Home.html")


@app.route('/Predict.html')
def predict():
    return render_template('Predict.html')


@app.route("/pred", methods=['post', 'get'])
def pred():
    x = [[float(x) for x in request.form.values()]]

    print(x)
    col = ['goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
    x = pd.DataFrame(x, columns=col)

    # print(x.shape)

    print(x)
    pred = model.predict(x)
    pred = le.inverse_transform(pred)
    print(pred[0])
    return render_template('Submit.html', prediction_text=str(pred))


if __name__ == "__main__":
    app.run(debug=False)
