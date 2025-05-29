# --- Imports ---
# Import the required libraries
import streamlit as st # for Web UI
from PyPDF2 import PdfReader # read all the PDF documents
from langchain.text_splitter import RecursiveCharacterTextSplitter # After reading pdf we need to convert into Chunk
import os
from dotenv import load_dotenv # To load Environment variable

from langchain_community.vectorstores import FAISS # Vector Embading store
from langchain.embeddings import OpenAIEmbeddings # For Embadding Model
from langchain.chat_models import ChatOpenAI # LLM model
from langchain.chains.question_answering import load_qa_chain # do the chat and prompt
from langchain.prompts import PromptTemplate

# --- Load environment variables ---
load_dotenv() # to load all the environment variables
os.getenv("OPENAI_API_KEY")  # Ensure this is set correctly

# --- PDF Text Extraction ---
# Get the PDF Text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    # Loop for multiple PDFs
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # Read the PDF using PdfReader library
        for page in pdf_reader.pages: # Loop for multiple page and read each page
            page_text = page.extract_text() # Extract all the text from each pages and store
            if page_text:
                text += page_text
    return text

# --- Text Chunking ---
# Get the chunks from the PDF text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) 
    chunks = text_splitter.split_text(text)
    return chunks # Return the chunks

# --- Create Vector Store ---
# Convert Chunks to  Vector
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Facebook AI Similarity Search
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# --- Build Conversational Chain ---
# Define prompt and context
def get_conversational_chain():
    prompt_template = """
    You are a smart chatbot and give the answer the question as detailed as possible using the context provided.
    If the answer is not in the context, reply:
    "answer is not available in the context - you can write an email to info@yourdomain.com or you can call on 070 XXX XXXX number for more information".
    Do not make up any answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
     # Generate Prompt based on the Prompt Template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Initialise the LLM Model
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    
    # Creating the chain of prompt 
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain 

# --- Handle User Input ---
# User input for asking the question
def user_input(user_question):
    # same embadding for Question
    # Loading the FAISS index from the local database
    # allow_dangerous_deserialization = We trust on the source of data (in this case we have generated knowledgebase)
    # The pickle module implements binary protocols for serializing and de-serializing a Python object
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # do the similarity search from knowledge based and user question
        docs = new_db.similarity_search(user_question)

        if not docs:
            st.warning("No relevant documents found in the vector store.")
            return
        # getting the chain
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])

    except Exception as e:
        st.error(f"Error loading or querying FAISS index: {e}")

# --- Streamlit Web App ---
def main():
    st.set_page_config("Chat with Multiple PDF")
    st.image('logo//YOUR_LOGO.png')
    st.header("RAG-based Chat App: Demo (Local)")
    
    user_question = st.text_input("Ask a Question from the uploaded PDF files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        # Upload the PDF to convert into Vector Embadding...!
        st.header("Upload PDFs for Vector Embedding")
        pdf_docs = st.file_uploader("Upload and process PDF files", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs) # Getting PDF text
                    text_chunks = get_text_chunks(raw_text) # Getting the chunks
                    get_vector_store(text_chunks) # get the vector store
                    st.success("Documents processed and saved to FAISS index.")
            else:
                st.warning("Please upload at least one PDF.")

# --- Entry Point ---
if __name__ == "__main__":
    main()
