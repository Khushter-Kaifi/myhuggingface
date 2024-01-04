# Finacial Sentiment Analysis Using Huggingface App
# Team Name :- Free Thinkers
# Authors:- Lalit Chaudhary and Khushter Kaifi
# Update On- 2 Jan 2024

# streamlit is a Python library used for creating web applications with minimal effort.
# pipeline is a class from the Hugging Face Transformers library that allows you to easily use pre-trained models for various natural language processing (NLP) tasks

import streamlit as st
from transformers import pipeline

# This line creates a sentiment analysis pipeline using the Hugging Face Transformers library. 
# The pipeline is pre-configured to perform sentiment analysis on input text.
# # Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Sets the title of the Streamlit web application
st.title("Financial Sentiment Analysis Using HuggingFace \n Team Name:- Free Thinkers")

# Displays a text input box where the user can enter a sentence for sentiment analysis.
st.write("Enter a Sentence to Analyze the Sentiment:")
user_input = st.text_input("")
st.write("Press the Enter key")

# Performing Sentiment Analysis:
# Checks if the user has entered some text. If yes, 
# it uses the sentiment_pipeline to analyze the sentiment of the input text and stores the result in the result variable.

if user_input:
    result = sentiment_pipeline(user_input)
    sentiment = result[0]["label"]
    confidence = result[0]["score"]


# Displaying Results:
#If there is user input, it displays the sentiment and confidence score. 
# The sentiment is extracted from the "label" field in the result, and the confidence score is extracted from the "score" field.
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {confidence:.2%}")
