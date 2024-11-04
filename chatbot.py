# Import necessary libraries
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the conversational model and tokenizer, and move the model to the GPU if available
model_name = "microsoft/DialoGPT-small"  # You can try larger models like DialoGPT-medium if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Initialize conversation history in Streamlit session state
if "history" not in st.session_state:
    st.session_state.history = []

# Define a function to generate AI-related responses
def ai_chatbot_conversation(input_text):
    # Tokenize the input and previous conversation history
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt').to(device)
    
    # Convert history to a tensor if it exists; otherwise, just use the new input tensor
    if st.session_state.history:
        bot_input_ids = torch.cat([torch.cat(st.session_state.history, dim=-1), new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    # Generate a response from the model with adjusted settings
    output = model.generate(
        bot_input_ids,
        max_length=200,          # Increase max_length if responses are too short
        temperature=0.7,         # Add slight randomness for varied responses
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the output to get the response text
    response = tokenizer.decode(output[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Update the conversation history as a list of tensors
    st.session_state.history.append(new_user_input_ids)

    return response

# Streamlit app layout
st.title("AI Chatbot")
st.write("Hello! I can provide information about Artificial Intelligence. Type your message below and hit Enter.")

# Input box for user input
user_input = st.text_input("You: ", key="input")

# Display bot response if there is user input
if user_input:
    response = ai_chatbot_conversation(user_input)
    st.write("AI Chatbot:", response)

# Button to clear conversation history
if st.button("Clear Chat"):
    st.session_state.history = []
    st.write("Chat history cleared.")
