# SmartQueryBot
SmartQueryBot  is an AI-powered chatbot that refines user queries and provides accurate responses by leveraging Pinecone for semantic search and ChatGroq for query enhancement. It offers a seamless conversational experience with real-time context, built using Streamlit and advanced ML models.

**OVERVIEW:**

This project involves the development of an intelligent chatbot using a combination of advanced AI technologies to enhance user interaction. The chatbot utilizes Pinecone for efficient vector-based search and ChatGroq for query refinement. The application allows users to input queries, which are then refined to provide more accurate and relevant responses. The project leverages machine learning models to improve the chatbot's ability to understand context and provide precise answers from a knowledge base.

**Key features of the project:**

**Query Refinement:** The chatbot refines user queries using the ChatGroq API, enhancing the relevance and quality of responses.

**Contextual Understanding:** It maintains a conversation log to provide context for generating better responses, allowing the chatbot to engage in more natural and meaningful conversations.

**Pinecone Integration:** It uses Pinecone to retrieve the most relevant information from a knowledge base, improving the bot's accuracy and ability to handle complex queries.

**User Interface:** Built with Streamlit, the application provides a user-friendly interface that allows users to interact with the chatbot in real-time.

**Machine Learning Models:** Utilizes the SentenceTransformer model for encoding user inputs and finding semantic matches in the knowledge base.

**Tech Stack:**

**Pinecone:**

A vector database used for storing and querying high-dimensional vectors, helping with fast and scalable search functionality. It's used here to retrieve the most relevant answers from the knowledge base based on semantic similarity.

**ChatGroq API:**

A conversational AI tool used to refine user queries and generate improved responses. The ask() method from ChatGroq is used to process the conversation context and generate the most appropriate query.

**Streamlit:**

A Python-based framework used to build the frontend of the application. Streamlit allows for rapid prototyping of machine learning applications with an intuitive user interface for interacting with the chatbot.

**SentenceTransformer:**

A pre-trained model used for encoding text inputs into fixed-size vectors that capture semantic meaning. These vectors are then used to query Pinecone for the most similar matches, ensuring relevant responses are provided.

**Python:**

The primary programming language used to implement the logic of the chatbot, integrate different technologies, and run machine learning models.

**Environment Variables:**

The API keys for Pinecone and ChatGroq are securely stored and accessed through environment variables, ensuring that sensitive information is not hardcoded into the source code.

**Chatbot UI:**

The application is designed with a simple UI where users input their queries, and the chatbot responds based on the refined queries and the information retrieved from the knowledge base.

**Flow of Interaction:**

Step 1: The user enters a query in the Streamlit interface.

Step 2: The query is refined using the ChatGroq API, which processes the conversation context and generates an improved query.

Step 3: The refined query is then encoded into a vector using the SentenceTransformer model.

Step 4: The vectorized query is used to search the Pinecone index for the most relevant pieces of information.

Step 5: The chatbot responds with the best match retrieved from Pinecone or additional information based on the refined query.

This intelligent system continuously learns from interactions and improves the accuracy of its responses, offering users a more conversational experience with real-time feedback.
