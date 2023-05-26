import os 
from apikey import API_KEY
import streamlit as st 
from langchain.llms import OpenAI 
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = API_KEY


# App framework 
st.title('ONE800 Memory Feature Langchain test')
prompt = st.text_input('plug in your prompt here ')

# Prompt Templete: 
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'write me a youtube vidoe title about {topic}'
)


# Memory 
title_memory = ConversationBufferMemory(input_key = 'topic', 
                                        memory_key = 'chat_history')

# LLM
llm = OpenAI(temperature=0.9)

title_chain = LLMChain(llm = llm, 
                       prompt = title_template, 
                       output_key = 'title', 
                       memory = title_memory,
                       verbose = True) # verbose is able to showcase what is inputing into chatGPT 


if prompt: 
    title = title_chain.run(prompt)
    st.write(title) # print out the gpt response 