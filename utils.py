import os
import streamlit as st  # Add this import
from langchain_groq import ChatGroq
import pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone (with the provided API key directly in the code)
class PineconeHandler:
    def __init__(self, api_key, environment):
        self.pc = pinecone.Pinecone(api_key=api_key, environment=environment)

    def get_index(self, index_name):
        return self.pc.index(index_name)


# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pinecone API key and environment based on your provided key
PINECONE_API_KEY = 'pcsk_44ocaD_QVT9vJJfVPFxPVgVWqiDXCqqZ7Zx8Tw57neQuRn9BqfMyqcVaGm9NBEZn2ftsD'  # Use your provided key here
PINECONE_ENVIRONMENT = 'us-west1-gcp'  # Example environment, update this based on your configuration

# Initialize Pinecone
pinecone_handler = PineconeHandler(PINECONE_API_KEY, PINECONE_ENVIRONMENT)
INDEX_NAME = 'langchain-chatbot'

# Function to find a match in the Pinecone index
def find_match(input):
    input_em = model.encode(input).tolist()
    index = pinecone_handler.get_index(INDEX_NAME)
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

# Function to refine the query using ChatGroq API
def query_refiner(conversation, query):
    # Initialize the ChatGroq model with API key directly
    groqllm = ChatGroq(model="llama3-8b-8192", api_key="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")  # Use your provided key here

    try:
        # Use the correct method for the model, such as 'query()'
        response = groqllm.query(  # Replace 'ask' with 'query' or the correct method for your version
            prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response["choices"][0]["text"]
    except AttributeError as e:
        raise RuntimeError(f"Error using ChatGroq: {e}")
    except Exception as e:
        raise RuntimeError(f"Error while refining query: {e}")

# Function to generate the conversation string for context
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"
    return conversation_string
