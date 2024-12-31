# # from langchain.chains import ConversationChain
# # from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# # from langchain.prompts import (
# #     SystemMessagePromptTemplate,
# #     HumanMessagePromptTemplate,
# #     ChatPromptTemplate,
# #     MessagesPlaceholder
# # )
# # import streamlit as st
# # from streamlit_chat import message
# # from utils import *

# # from langchain_groq import ChatGroq  # Importing ChatGroq

# # groqllm = ChatGroq(
# #     model="llama3-8b-8192",
# #     api_key="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"
# # )

# # st.subheader("Chatbot with Langchain, ChatGroq, Pinecone, and Streamlit")

# # if 'responses' not in st.session_state:
# #     st.session_state['responses'] = ["How can I assist you?"]

# # if 'requests' not in st.session_state:
# #     st.session_state['requests'] = []

# # if 'buffer_memory' not in st.session_state:
# #     st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# # system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
# # and if the answer is not contained within the text below, say 'I don't know'""")

# # human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

# # prompt_template = ChatPromptTemplate.from_messages([
# #     system_msg_template,
# #     MessagesPlaceholder(variable_name="history"),
# #     human_msg_template
# # ])

# # conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=groqllm, verbose=True)

# # # container for chat history
# # response_container = st.container()
# # # container for text box
# # textcontainer = st.container()

# # with textcontainer:
# #     query = st.text_input("Query: ", key="input")
# #     if query:
# #         with st.spinner("typing..."):
# #             conversation_string = get_conversation_string()
# #             # st.code(conversation_string)
# #             refined_query = query_refiner(conversation_string, query)
# #             st.subheader("Refined Query:")
# #             st.write(refined_query)
# #             context = find_match(refined_query)
# #             # print(context)  
# #             response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
# #         st.session_state.requests.append(query)
# #         st.session_state.responses.append(response) 
# # with response_container:
# #     if st.session_state['responses']:

# #         for i in range(len(st.session_state['responses'])):
# #             message(st.session_state['responses'][i], key=str(i))
# #             if i < len(st.session_state['requests']):
# #                 message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')



# import os
# from langchain_groq import ChatGroq
# import pinecone
# from sentence_transformers import SentenceTransformer

# # Initialize Pinecone (with the provided API key directly in the code)
# class PineconeHandler:
#     def __init__(self, api_key, environment):
#         self.pc = pinecone.Pinecone(api_key=api_key, environment=environment)

#     def get_index(self, index_name):
#         return self.pc.index(index_name)


# # Initialize SentenceTransformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Pinecone API key and environment based on your provided key
# PINECONE_API_KEY = 'pcsk_44ocaD_QVT9vJJfVPFxPVgVWqiDXCqqZ7Zx8Tw57neQuRn9BqfMyqcVaGm9NBEZn2ftsD'  # Use your provided key here
# PINECONE_ENVIRONMENT = 'us-west1-gcp'  # Example environment, update this based on your configuration

# # Initialize Pinecone
# pinecone_handler = PineconeHandler(PINECONE_API_KEY, PINECONE_ENVIRONMENT)
# INDEX_NAME = 'langchain-chatbot'

# # Function to find a match in the Pinecone index
# def find_match(input):
#     input_em = model.encode(input).tolist()
#     index = pinecone_handler.get_index(INDEX_NAME)
#     result = index.query(input_em, top_k=2, includeMetadata=True)
#     return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

# # Function to refine the query using ChatGroq API
# def query_refiner(conversation, query):
#     # Initialize the ChatGroq model with API key directly
#     groqllm = ChatGroq(model="llama3-8b-8192", api_key="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")  # Use your provided key here

#     try:
#         # Use the correct method (query or another based on available methods)
#         response = groqllm.query(  # Replace 'ask' with 'query' or correct method
#             prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
#             temperature=0.7,
#             max_tokens=256,
#             top_p=1,
#             frequency_penalty=0,
#             presence_penalty=0
#         )
#         return response["choices"][0]["text"]
#     except AttributeError as e:
#         raise RuntimeError(f"Error using ChatGroq: {e}")
#     except Exception as e:
#         raise RuntimeError(f"Error while refining query: {e}")

# # Function to generate the conversation string for context
# def get_conversation_string():
#     conversation_string = ""
#     for i in range(len(st.session_state['responses'])-1):
#         conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
#         conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"
#     return conversation_string


from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
from langchain_groq import ChatGroq  # Importing ChatGroq

# Initialize the ChatGroq model with API key directly
groqllm = ChatGroq(
    model="llama3-8b-8192",
    api_key="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"  # Use your provided key here
)

st.subheader("Chatbot with Langchain, ChatGroq, Pinecone, and Streamlit")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([ 
    system_msg_template,
    MessagesPlaceholder(variable_name="history"),
    human_msg_template
])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=groqllm, verbose=True)

# Container for chat history
response_container = st.container()
# Container for text input box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)  # Uses updated query_refiner
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            try:
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            except Exception as e:
                st.error(f"Error while generating response: {e}")
                response = "Sorry, I couldn't process your request."
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
