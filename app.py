from flask import Flask, render_template, request
import pickle
import numpy as np
model = pickle.load(open("model.pkl", "rb"))
app = Flask(__name__)


@app.route('/')
def man():
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    arr = np.array([[data1, data2, data3]])
    pre = model.predict(arr)
    output = round(pre[0], 2)
    return render_template("home.html", prediction_text = "PREDICTION : Rs.{}".format(output))
if __name__ == "__main__":
    app.run(debug=True)
