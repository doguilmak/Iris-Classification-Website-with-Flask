from flask import Flask, render_template, redirect, url_for, request
import pickle

app = Flask(__name__)

class_map = {0: "Iris-setosa", 
            1: "Iris-versicolor", 
            2: "Iris-virginica"}

def predict_class(sepal_length, sepal_width, petal_length, petal_width):

    # Read the machine learning model
    pickle_file = open('classifier.pkl', 'rb')     
    classifier = pickle.load(pickle_file)

    y_predict = classifier.predict([[sepal_length, 
                                     sepal_width, 
                                     petal_length, 
                                     petal_width]])[0]

    return class_map[y_predict]


@app.route("/", methods = ['POST', 'GET'])
def index():
    irisClass=''
    if request.method == 'POST' and 'sepal_length_input' in request.form and 'sepal_width_input' in request.form and 'petal_length_input' in request.form and 'petal_width_input' in request.form:
        sepal_length = float(request.form.get('sepal_length_input'))
        sepal_width = float(request.form.get('sepal_width_input'))
        petal_length = float(request.form.get('petal_length_input'))
        petal_width = float(request.form.get('petal_width_input'))
        irisClass = predict_class(sepal_length, sepal_width, petal_length, petal_width)
    
    return render_template("index.html", irisClass=irisClass)

if __name__ == "__main__":
    app.run(debug=True)
