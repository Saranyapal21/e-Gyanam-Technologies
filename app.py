# import model_initialize
from run_llm import run_llm
import streamlit as st
from streamlit_chat import message

st.header("Ramayana Chatbot")

prompt = st.text_input("Prompt", placeholder="Enter your question about Ramayana here")

if prompt:
    with st.spinner("Generating response...."):
        generated_response = run_llm(query = prompt)
        message(generated_response['result'], is_user=True)
        # message