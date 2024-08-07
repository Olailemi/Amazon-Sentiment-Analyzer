import requests  # Library for making HTTP requests

def get_sentiment(text):
    """
    Sends a POST request to a local sentiment analysis API and returns the sentiment result.
    
    Parameters:
    text (str): The text to analyze.
    
    Returns:
    dict: Sentiment analysis result if the request is successful.
    None: If the request fails.
    """
    url = 'http://127.0.0.1:8000/analyze'  # Local API endpoint
    payload = {'text': text}  # Data to send to the API
    response = requests.post(url, json=payload)  # Send POST request with JSON payload
    
    if response.status_code == 200:  # Check if request was successful
        return response.json()  # Return the JSON response from the API
    else:
        print(response.status_code)  # Print status code if request failed
        
if __name__ == "__main__":
    text = input("Enter text to analyze: ")  # Get input from the user
    sentiment = get_sentiment(text)  # Analyze the sentiment of the input text
    
    print(sentiment)  # Print the result of the sentiment analysis
