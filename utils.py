import os
import streamlit as st
from langchain_groq import ChatGroq
import pinecone
from sentence_transformers import SentenceTransformer

class PineconeHandler:
    def __init__(self, api_key, environment):
        self.pinecone = pinecone.Pinecone(api_key=api_key)
        
    def get_index(self, index_name):
        return self.pinecone.Index(index_name)

# Initialize components
model = SentenceTransformer('all-MiniLM-L6-v2')
PINECONE_API_KEY = 'pcsk_6btsAL_AKwUz13ZAN6zR2zQ83EzNhkEWqvfon2ofhGYzebMGuvZdaifFJyZjvZK6KrwZ5m'
PINECONE_ENVIRONMENT = 'us-east-1'
INDEX_NAME = 'democracy-chatbot'

pinecone_handler = PineconeHandler(PINECONE_API_KEY, PINECONE_ENVIRONMENT)

def find_match(input):
    try:
        input_em = model.encode(input).tolist()
        index = pinecone_handler.get_index(INDEX_NAME)
        result = index.query(vector=input_em, top_k=2, include_metadata=True)
        
        if not result['matches']:
            return "No relevant information found."
            
        matches = result['matches']
        context = []
        
        for match in matches:
            if 'metadata' in match and 'text' in match['metadata']:
                context.append(match['metadata']['text'])
                
        return "\n".join(context) if context else "No text content found."
        
    except Exception as e:
        st.error(f"Error querying Pinecone: {str(e)}")
        return "Error retrieving information."

def query_refiner(conversation, query):
    groqllm = ChatGroq(
        model_name="llama3-70b-8192",
        groq_api_key="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"
    )
    
    prompt = f"""Given the following conversation and query, refine the query to be most relevant for knowledge base search.
    
    Conversation: {conversation}
    Query: {query}
    
    Refined query:"""
    
    response = groqllm.invoke(prompt)
    return response.content

def get_conversation_string(responses, requests):
    conversation_string = ""
    for i in range(len(responses)-1):
        conversation_string += f"Human: {requests[i]}\n"
        conversation_string += f"Assistant: {responses[i+1]}\n"
    return conversation_string
