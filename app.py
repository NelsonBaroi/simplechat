from flask import Flask, render_template, request, jsonify
import nltk
import json
from transformers import pipeline

# Download necessary NLTK data files
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load Hugging Face zero-shot-classification model
classifier = pipeline("zero-shot-classification")

# Load intents from the JSON file
def load_intents(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

intents = load_intents('intents.json')  # Adjust the path if necessary

# Function to classify intent based on keyword matching
def classify_intent(user_input, intents):
    for intent in intents:
        for prompt in intent['prompt']:
            if user_input.lower() == prompt.lower():
                return intent['completion']
    return None

# Function to get response from Hugging Face zero-shot-classification model
def get_ai_response(question, classifier):
    try:
        result = classifier(
            question,
            candidate_labels=[
                'greeting',
                'professional background',
                'leadership',
                'contact',
                'location',
                'definition',
            ],
        )
        return result['labels'][0]  # Returns the label with highest score
    except Exception as e:
        print(f"Error fetching AI response: {str(e)}")
        return None

# Initialize Flask app
app = Flask(__name__)

# Route to serve the homepage
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint to handle chatbot responses
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message'].strip()

    # Exit conditions
    if user_input.lower() in ["exit", "quit", "bye"]:
        return jsonify({'response': "Goodbye!"})

    # Try to get a response based on predefined intents
    response = classify_intent(user_input, intents)
    if response:
        return jsonify({'response': response})
    
    # Use AI classification for more complex queries
    ai_response = get_ai_response(user_input, classifier)
    if ai_response == 'greeting':
        response = "Hello! How can I assist you today?"
    elif ai_response == 'contact':
        response = "You can contact me at nelson6114007@gmail.com or +8801786068822."
    elif ai_response == 'location':
        response = "I live in Pabna, Bangladesh."
    elif ai_response == 'definition':
        response = "I am an AI assistant. I can help you with information and tasks."
    elif ai_response:
        response = "Sorry, I didn't quite understand that. Could you ask something else?"
    else:
        response = "Sorry, I am unable to process your request."

    return jsonify({'response': response})

# Run the Flask app
if __name__ == '__main__':
    from os import environ
    app.run(host="0.0.0.0", port=int(environ.get("PORT", 10000)))