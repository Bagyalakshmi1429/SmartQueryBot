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
from utils import find_match, query_refiner, get_conversation_string
from langchain_groq import ChatGroq

def main():
    groqllm = ChatGroq(
        model_name="llama3-70b-8192",
        groq_api_key="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"
    )

    st.subheader("Chatbot with Langchain, ChatGroq, Pinecone, and Streamlit")

    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []
    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    system_msg_template = SystemMessagePromptTemplate.from_template(
        """Answer the question as truthfully as possible using the provided context, 
        and if the answer is not contained within the text below, say 'I don't know'"""
    )

    human_msg_template = HumanMessagePromptTemplate.from_template("{input}")

    prompt_template = ChatPromptTemplate.from_messages([
        system_msg_template,
        MessagesPlaceholder(variable_name="history"),
        human_msg_template
    ])

    conversation = ConversationChain(
        memory=st.session_state.buffer_memory,
        prompt=prompt_template,
        llm=groqllm,
        verbose=True
    )

    response_container = st.container()
    textcontainer = st.container()

    with textcontainer:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("typing..."):
                try:
                    conversation_string = get_conversation_string(
                        st.session_state['responses'],
                        st.session_state['requests']
                    )
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

    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i], key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

if __name__ == "__main__":
    main()
