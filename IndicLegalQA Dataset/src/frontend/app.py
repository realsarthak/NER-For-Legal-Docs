from flask import Flask, render_template, request, session
from datetime import datetime
import spacy

app = Flask(__name__)
app.secret_key = 'd'  # Change this for production

# Load your trained model
nlp = spacy.load("/Users/sarthak/Desktop/NLTP TA3/IndicLegalQA Dataset/model_output/model/model-best")

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'history' not in session:
        session['history'] = []
    
    entities = []
    if request.method == 'POST':
        text = request.form['text']
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Add to history
        session['history'].append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'text': text,
            'entities': entities
        })
        session.modified = True
    
    return render_template('index.html', 
                         entities=entities,
                         history=session.get('history', []))

if __name__ == '__main__':
    app.run(debug=True)