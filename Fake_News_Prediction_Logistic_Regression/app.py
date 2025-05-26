from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

port_stem = PorterStemmer()
nltk.download('stopwords')

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# @app.route("/",methods=["GET", "POST"])
# def home():
#     return render_template("index.html")

# @app.route('/predict',methods = ["POST"])
# def predict():
#     if request.method == 'POST':
#         title = request.form['title']
#         url = request.form['url']
#         source = request.form['source']
#         tweets = request.form['tweets']
#         combined_text = f"{title} {url} {source} {tweets}"
#         combined_text = stemming(combined_text)
#         transformed_input = vectorizer.transform([combined_text])
#         pred = model.predict(transformed_input)[0]
#         print("Prediction:", pred)
#         prediction = "Fake News ❌" if pred == 1 else "Real News ✅"
#     return render_template('index.html', prediction=prediction)  

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        # Get form data
        title = request.form.get('title', '')
        url = request.form.get('url', '')
        source = request.form.get('source', '')
        tweets = request.form.get('tweets', '')
        
        # Combine all text
        combined_text = f"{title} {url}"
        
        # Preprocess the text
        processed_text = stemming(combined_text)
        
        # Make prediction
        transformed_input = vectorizer.transform([processed_text])
        pred = model.predict(transformed_input)[0]
        
        print("Prediction:", pred)
        prediction = "Fake News ❌" if pred == 0 else "Real News ✅"
        
        return render_template('index.html', prediction=prediction)
    
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)