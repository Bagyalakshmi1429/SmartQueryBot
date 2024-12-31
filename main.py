from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
from streamlit_chat import message
from utils import find_match, query_refiner
from langchain_groq import ChatGroq

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += f"Human: {st.session_state['requests'][i]}\n"
        conversation_string += f"Assistant: {st.session_state['responses'][i+1]}\n"
    return conversation_string

# Initialize Groq
groqllm = ChatGroq(
    model_name="llama3-70b-8192",
    groq_api_key="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"
)

st.subheader("Chatbot with Langchain, ChatGroq, Pinecone, and Streamlit")

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Set up prompt templates
system_msg_template = SystemMessagePromptTemplate.from_template("""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")

human_msg_template = HumanMessagePromptTemplate.from_template("{input}")

prompt_template = ChatPromptTemplate.from_messages([
    system_msg_template,
    MessagesPlaceholder(variable_name="history"),
    human_msg_template
])

# Initialize conversation chain
conversation = ConversationChain(
    memory=st.session_state.buffer_memory, 
    prompt=prompt_template, 
    llm=groqllm, 
    verbose=True
)

# Create containers
response_container = st.container()
textcontainer = st.container()

# Handle user input
with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            try:
                conversation_string = get_conversation_string()
                refined_query = query_refiner(conversation_string, query)
                st.subheader("Refined Query:")
                st.write(refined_query)
                
                context = find_match(refined_query)
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                
                st.session_state.requests.append(query)
                st.session_state.responses.append(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                response = "Sorry, I couldn't process your request."

# Display chat history
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
