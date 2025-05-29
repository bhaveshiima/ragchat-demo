# ragchat-demo
RAG-based Chatbot

In this demo, we need Python 3.12 environment and OPENAI_API_KEY

There are two phase to creating RAG chatbot 

# Create a Knowledge base
- Uploading document
- Splitting
- Chunking 
- Embedding 
- Vector store

# Create a Chatbot (User Interface)
- Developing UI
- Embedding of Question
- Similarity search (Cosine similarity search)

# Tools and framework used in this local RAG Application
- Google Gemini / OpenAI model: for embedding the model.
- LangChain: for splitting/chunking the documents.
- FAISS: we store the vector data after generating the index.
- Google Gemini Pro LLM model / OpenAI o4-mini Model: to generate the response based on the question.
- Streamlit: For the web user interface, users can ask questions.
- Python: Python language for creating an application
