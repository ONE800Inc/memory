# bring in deps 

import os 
from langchain_testings.apikey import API_KEY

import streamlit as st 
from langchain.llms import OpenAI 

os.environ['OPEN_AI_KEY'] = API_KEY

st.title('Youtube GPT Creator')
prompt = st.text_input('plug in your prompt here ')