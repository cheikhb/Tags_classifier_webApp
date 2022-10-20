import flask
import pickle

# Use pickle to load in the pre-trained model.
with open(f'model/countvec.pkl', 'rb') as file:
    count_vectorizer = pickle.load(file)
    
with open(f'model/svm_clf.pkl', 'rb') as file:
    classifier = pickle.load(file)

def predict_category(text):
    result = classifier.predict(count_vectorizer.transform([text]))
    return(result[0])
    

app = flask.Flask(__name__, template_folder='templates')
    
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('templates/main.html'))
    if flask.request.method == 'POST':
        news_text = ' '
        news_text = flask.request.form['news_text']
        prediction = predict_category(news_text)
        return flask.render_template('templates/main.html',
                                     original_input={'News Text':news_text},
                                     result=prediction)
        
if __name__ == '__main__':
    app.run()