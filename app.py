import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
import openai
import os

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

# Streamlit UI
st.title("RAG-based Q&A from URLs")
url1 = st.text_input("Enter first URL:")
url2 = st.text_input("Enter second URL:")
process_button = st.button("Process URLs")

if process_button and url1 and url2:
    st.write("Fetching and processing content...")
    
    # Extract content from URLs
    text1 = extract_text_from_url(url1)
    text2 = extract_text_from_url(url2)
    
    #Create chunks of the contents
    documents = [Document(page_content=text1), Document(page_content=text2)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    #Generate embeddings
    embeddings = OpenAIEmbeddings()
    # vectorstore = FAISS.from_documents(split_docs, embeddings)
    # Create Chroma vectorstore (persistent storage in "./chroma_db")
    vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()
    
    # Give the system prompt
    custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI assistant that answers questions based on the provided documents.

    Context:
    {context}

    Question: {question}

    Provide a **concise and accurate** answer. If the answer is not found in the context, say "I don't know."
    """
)
    
    # Define the retrieval chain
    llm = ChatOpenAI(model_name="gpt-4o-mini")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, 
                                           chain_type_kwargs={"prompt": custom_prompt} , return_source_documents=True,  verbose=True)
    
    st.session_state["qa_chain"] = qa_chain
    st.success("Processing complete! You can now ask questions.")

# Question Answering
if "qa_chain" in st.session_state:
    question = st.text_input("Ask a question based on the content:")
    if st.button("Get Answer") and question:
        result = st.session_state["qa_chain"]({"query": question})

        # To debug on retrieved documents
        answer = result["result"]
        retrieved_docs = result["source_documents"]
        st.write("**Answer:**", answer)