import os
import streamlit as st
from dotenv import load_dotenv
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain.schema import AIMessage, SystemMessage, HumanMessage
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
MODEL_ID = "arn:aws:bedrock:us-east-1:303031092167:inference-profile/us.meta.llama3-1-70b-instruct-v1:0"
PDF_FILE_PATH = "./aws_migration_whitepaper.pdf"

# Constants
SYSTEM_PROMPT = """You are a helpful AWS cloud migration assistant. 
Use only the following context to answer the user's queries. 
If you are unsure, say so clearly."""

# Load and split documents
def load_and_split_documents(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(pages)
    except Exception as e:
        st.error(f"Error loading or splitting documents: {e}")
        return []

# Create vector store
@st.cache_resource
def create_vector_store(file_path):
    try:
        documents = load_and_split_documents(file_path)
        embeddings = BedrockEmbeddings(
            credentials_profile_name="jerry-bedrock",
            region_name="us-east-1",
            model_id="amazon.titan-embed-text-v2:0"
        )
        vector_store = FAISS.from_texts([doc.page_content for doc in documents], embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Retrieve relevant documents
def retrieve_relevant_docs(vector_store, query, k=3):
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return []

# Initialize the chat model
def initialize_chat_model(callback_handler):
    try:
        llm = ChatBedrock(
            model_id=MODEL_ID,
            provider="meta",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region='us-east-1',
            streaming=True,
            callbacks=[callback_handler]
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing chat model: {e}")
        return None

# Display chat messages
def display_chat_messages():
    for i, msg in enumerate(st.session_state["messages"]):
        if msg.type == "system":
            continue
        if i == len(st.session_state["messages"]) - 1 and msg.type == "assistant":
            # Skip the last assistant message; it's being handled in handle_user_input
            continue
        st.chat_message(msg.type).write(msg.content)

# Handle user input and generate response
def handle_user_input(query, vector_store):
    user_msg = HumanMessage(content=query)
    st.session_state["messages"].append(user_msg)
    st.chat_message("user").write(query)
    
    retrieved_docs = retrieve_relevant_docs(vector_store, query)
    context = " ".join([doc.page_content for doc in retrieved_docs])
    
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser Query: {query}\nAnswer:"
    
    # Create a chat message container for the assistant
    assistant_container = st.chat_message("assistant")
    
    # Create an empty placeholder inside the assistant container
    placeholder = assistant_container.empty()
    
    # Initialize the callback handler with the placeholder
    callback_handler = StreamlitCallbackHandler(placeholder)
    
    # Initialize the llm with the callback handler and streaming enabled
    llm = initialize_chat_model(callback_handler)
    
    # Generate response
    response_msg = llm.invoke(prompt)
    
    # Update the placeholder with the full response
    placeholder.write(response_msg.content)
    
    # Append the AI message to session state
    st.session_state["messages"].append(AIMessage(content=response_msg.content))

# Main function
def main():
    st.title("💬 AI Assistant on CDD")
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            SystemMessage(content="You are a helpful AI assistant."),
            AIMessage(content="How can I help you?")
        ]
    
    # Create vector store
    vector_store = create_vector_store(PDF_FILE_PATH)
    if vector_store is None:
        st.error("Failed to create vector store. Please check the logs.")
        return
    
    # Display existing messages
    display_chat_messages()
    
    # Wait for new user input
    if query := st.chat_input():
        handle_user_input(query, vector_store)

if __name__ == "__main__":
    main()