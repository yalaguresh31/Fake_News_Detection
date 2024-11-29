# from flask import Flask, request, render_template
# from markupsafe import escape
# import pickle



# vector = pickle.load(open("vectorizer.pkl",'rb'))
# model = pickle.load(open( "finalized_model.pkl",'rb'))

# app = Flask( __name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/prediction',methods=['GET','POST'])
# def prediction():
#     if request.method == 'POST':
#         news = str(request.form['news'])
#         print(news)
#         predict = model.predict(vector.transform([news]))[0]
#         print(predict)

#         return render_template('prediction.html',prediction_text="News headline is -> {}".format(predict))
#     else:
#         return render_template('prediction.html')
    
# if __name__ == '__main__':
#     app.run()

from flask import Flask, request, render_template
import pickle

# Load the trained model and vectorizer
vector = pickle.load(open("vectorizer.pkl", 'rb'))
pac_model = pickle.load(open("pac_model.pkl", 'rb'))
logistic_model = pickle.load(open("logistic_model.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Fetch the input news headline from the form
        news = str(request.form['news'])
        
        # Perform prediction using the trained models
        pac_predict = pac_model.predict(vector.transform([news]))[0]
        logistic_predict = logistic_model.predict(vector.transform([news]))[0]
        print(pac_predict)
        print(logistic_predict)
        # Map predictions from 0, 1 to 'Fake' and 'Real'
        pac_result = "Real" if pac_predict == "REAL" else "Fake"
        logistic_result = "Real" if logistic_predict == "REAL" else "Fake"
        
        # Render prediction result page and pass the prediction results
        return render_template('prediction.html', pac_result=pac_result, logistic_result=logistic_result)
    
    return render_template('prediction.html', pac_result="", logistic_result="")

if __name__ == '__main__':
    app.run(debug=True)
