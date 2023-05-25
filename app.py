# bring in deps 

import os 
from apikey import API_KEY

import streamlit as st 
from langchain.llms import OpenAI 
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = API_KEY

# Prompt Templete: 
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'write me a youtube vidoe title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = 'write me a youtube script based on this title TITLE: {title} while leveraging this wikipedia research: {wikipedia_research}'
)


# App framework 
st.title('ONE800 Memory Feature Langchain test')
prompt = st.text_input('plug in your prompt here ')


# Memory 
title_memory = ConversationBufferMemory(input_key = 'topic', 
                                        memory_key = 'chat_history')

script_memory = ConversationBufferMemory(input_key = 'title', 
                                        memory_key = 'chat_history')

# LLMs 
llm = OpenAI(temperature=0.9)

title_chain = LLMChain(llm = llm, 
                       prompt = title_template, 
                       output_key = 'title', 
                       memory = title_memory,
                       verbose = True) # verbose is able to showcase what is inputing into chatGPT 

script_chain = LLMChain(llm = llm, 
                        prompt = script_template, 
                        output_key = 'script',
                        memory = script_memory,
                        verbose = True)

# sequencial_chain = SequentialChain(chains = [title_chain, script_chain], 
#                                    input_variables = ['topic'], 
#                                    output_variables = ['title', 'script'], 
#                                    verbose=True)

wiki = WikipediaAPIWrapper()

if prompt: 
    # response = sequencial_chain({'topic': prompt})
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title = title, wikipedia_research = wiki_research)
    # st.write(response['title']) # print out the gpt response 
    # st.write(response['script'])
    st.write(title) # print out the gpt response 
    st.write(script)

    with st.expander('Title History: '): # for debugging purposes 
        st.info(title_memory.buffer)

    with st.expander('Script History: '): # for debugging purposes 
        st.info(script_memory.buffer)

    with st.expander('Wiki Research History: '): # for debugging purposes 
        st.info(wiki_research)
