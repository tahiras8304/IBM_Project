from fastapi import FastAPI
import joblib

app = FastAPI()

# Load the pre-trained model and vectorizer
model = joblib.load('text_classification_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Test route for root path
@app.get("/")
async def root():
    return {"message": "Welcome to the text classification API!"}

# POST route for predictions
@app.post("/predict")
async def predict(data: dict):
    # Extract the text from the incoming request
    text = data['text']
    
    # Vectorize the text using the loaded vectorizer
    text_tfidf = vectorizer.transform([text])
    
    # Make a prediction using the loaded model
    prediction = model.predict(text_tfidf)
    
    # Return the prediction result
    return {"prediction": prediction[0]}

