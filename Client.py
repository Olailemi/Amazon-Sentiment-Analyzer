from fastapi import FastAPI  # Web framework for building APIs
from pydantic import BaseModel  # Data validation and settings management
import nest_asyncio  # Patch to allow asyncio to run in environments with existing event loops
import uvicorn  # ASGI server for running FastAPI applications
import joblib  # Library for serialization and deserialization of Python objects

# Apply nest_asyncio to allow FastAPI to run in environments with an existing event loop (e.g., Jupyter notebook)
nest_asyncio.apply()

# Load the pre-trained SVM model pipeline from the specified file path
model = joblib.load(r"C:\Users\hp\Desktop\Team Bravo Project\Jupiter Notebook (Sentiment Analysis)\TBsentiment_mod.pkl")

app = FastAPI()  # Initialize the FastAPI application

class SentimentRequest(BaseModel):
    """
    Define the request model for the /analyze endpoint.
    
    Attributes:
    text (str): The text to analyze for sentiment.
    """
    text: str

@app.post("/analyze")
async def analyze(request: SentimentRequest):
    """
    Analyze the sentiment of the given text.
    
    Parameters:
    request (SentimentRequest): The request payload containing the text to analyze.
    
    Returns:
    dict: A dictionary with the sentiment result.
    """
    text = request.text  # Extract text from the request
    processed_text = [text]  # Prepare text for prediction

    # Predict sentiment using the loaded model
    prediction = model.predict(processed_text)

    # Create a response dictionary with the prediction result
    sentiment = {'sentiment': prediction[0]}  # Adjust based on the model's output

    return sentiment

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
