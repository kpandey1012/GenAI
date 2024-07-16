import streamlit as st
from pymongo import MongoClient
import hashlib
import os
import utils
from langchain_community.llms import ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import (
    CSVLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader, 
    PyPDFLoader, UnstructuredFileLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import PromptTemplate
import os
import json

# Constants and setup
folder_path = "chromadb"
cached_llm = ChatOllama(model="phi3", temperature=0.1)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=100, separators=["\n\n", "\n", "", ""]
)
embedding = OllamaEmbeddings(model="mxbai-embed-large")

FILE_TYPE_MAPPING = {
    ".csv": CSVLoader,
    ".txt": TextLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".html": UnstructuredHTMLLoader,
}

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["ragproj1"]
users_collection = db["credentials"]
chat_histories_collection = db["chat_histories"]
feedback_collection = db["feedback"]


# Function to fetch chat history for the current user
def fetch_user_chat_history(userid):
    user_history = chat_histories_collection.find({"SessionId": userid}, {"History": 1,"_id": 0})
    return list(user_history)


def fetch_chat_interactions(userid):  
    return list(chat_histories_collection.find({"SessionId": userid}, {"_id": 0}))


def fetch_feedback(userid):
    return list(feedback_collection.find({"userid": userid}, {"_id": 0}))


# Function to delete chat history for the current user
def delete_user_chat_history(userid):
    result = chat_histories_collection.delete_many({"SessionId": userid})
    return result.deleted_count


# Function to fetch feedback for the current user
def fetch_user_feedback(userid):
    user_feedback = feedback_collection.find({"userid": userid})
    return list(user_feedback)


# Function to store user feedback
def store_feedback(userid, feedback):
    feedback_collection.insert_one({
        "userid": userid,
        "feedback": feedback
    })


# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Function to validate user credentials
def validate_login(username, password):
    user = users_collection.find_one({"userid": username})
    if user and user["password"] == hash_password(password):
        return user["role"]
    return None


# Function to create a new user
def create_user(username, password, userrole):
    if users_collection.find_one({"userid": username}):
        return "User already exists"
    users_collection.insert_one({
        "userid": username,
        "password": hash_password(password),
        "role": userrole
    })
    return "User created successfully"


# Simulated functions for document upload and chat interaction
def upload_document(document):
    filename = document.name
    file_extension = os.path.splitext(filename)[1].lower()

    loader_class = FILE_TYPE_MAPPING.get(file_extension)
    if not loader_class:
        return {"error": "Unsupported file type"}

    # Save the uploaded file locally
    with open(os.path.join("files", filename), "wb") as f:
        f.write(document.getbuffer())

    # Load and split documents
    loader = loader_class(os.path.join("files", filename))
    docs = loader.load_and_split()
    chunks = text_splitter.split_documents(docs)

    # Vectorization and storage
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    # Response details
    response = {
        "status": "Successfully Uploaded",
        "filename": filename,
        "doc_len": len(docs),
        "chunks_len": len(chunks)
    }
    return response


def query_chatbot(query, userid):
    vector_store = Chroma(
        persist_directory=folder_path, embedding_function=embedding
    )
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 1}
    )

    contextualize_q_system_prompt = (
        """Given a chat history and the latest user question, which might reference context in the chat history,
        formulate a standalone question which can be understood without the chat history. Do NOT answer the question,
        Just reformulate it if needed and otherwise return it as is."""
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    history_aware_retriever = create_history_aware_retriever(cached_llm, retriever, contextualize_q_prompt)

    system_prompt = (
         """You are an intelligent assistant, helping User by following directives and answering questions.
        Generate your response by following the steps below:
        1. Recursively break down the user input into smaller questions/directives
        2. For each atomic question/directive:
        3. Generate a draft response using the selected information
        4. Remove duplicate content from the draft response
        5. Generate your final response after adjusting it to increase accuracy and relevance
        6. Now only show your final response! Do not provide any explanations or details.
        "\n\n"
        {context}
        """
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    question_answer_chain = create_stuff_documents_chain(cached_llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id):
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string="mongodb://localhost:27017",
            database_name="ragproj1",
            collection_name="chat_histories"
        )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    llm_response = conversational_rag_chain.invoke({"input": query}, config={"configurable": {"session_id": userid}})

    sources = []
    for doc in llm_response["context"]:
        sources.append({"source": doc.metadata["source"], "page_content": doc.page_content})

    print(llm_response["answer"])
    #print(json.loads(llm_response["answer"]))

    response_answer = {
        "answer": (llm_response["answer"]),
        "sources": sources
    }

    return response_answer, 200





# Streamlit UI
st.sidebar.title("Capstone Group 3 BOT")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['user_role'] = None

if not st.session_state['logged_in']:
    # Login Section
    st.sidebar.header("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")

    if login_button:
        user_role = validate_login(username, password)
        if user_role:
            st.session_state['user_role'] = user_role
            st.session_state['sessionid'] = username
            st.session_state['logged_in'] = True
            st.sidebar.success(f"Logged in as {user_role}")
        else:
            st.sidebar.error("Invalid username or password")
else:
    # Logout Option
    st.sidebar.header(f"Welcome, {st.session_state['user_role']}")
    logout_button = st.sidebar.button("Logout")
    if logout_button:
        st.session_state['logged_in'] = False
        st.session_state['user_role'] = None
        st.rerun()  # This refreshes the app to the login screen

    # Display based on user role
    if st.session_state['user_role'] == "end_user":    
        # Sidebar actions
        st.sidebar.subheader("Actions")
        action = st.sidebar.selectbox("Select an action", ["chat","View Chat History", "Delete Chat History", "Provide Feedback"])
        if action == "chat":
            st.title("Chat Bot")
            query = st.chat_input(placeholder="your query here!")
            sessionid = st.session_state['sessionid']
            if query :            
                response, status_code = query_chatbot(query, sessionid)
                if status_code == 200:
                    st.subheader("Answer:")
                    st.write(response["answer"])

                    # Display source with toggle for additional context
                    st.subheader("Source:")
                    with st.expander("Click to show source"):
                        st.write(response["sources"])

                else:
                    st.error("Failed to get response from chatbot.")
        elif action == "View Chat History":
            print(st.session_state['sessionid'])
            user_chat_history = fetch_user_chat_history(st.session_state['sessionid'])
            if user_chat_history:
                st.subheader("Chat History")
                for chat in user_chat_history:
                    #st.write(f"Query: {type(chat)}")
                    chatuser = json.loads(chat['History'])
                    if chatuser['type'] == 'human':
                        st.write(f"Query: {chatuser['data']['content']}")
                        #st.write(f"{type(chatuser['data']['content'])}")
                    if chatuser['type'] == 'ai':
                        st.write(f"Response: {(chatuser['data']['content'])}")
                        #st.write(f": {type(chatuser['data']['content'])}")
            else:
                st.write("No chat history available.")

        elif action == "Delete Chat History":
            if st.button("Delete Chat History"):
                deleted_count = delete_user_chat_history(st.session_state['sessionid'])
                if deleted_count > 0:
                    st.success("Chat history deleted successfully.")
                else:
                    st.warning("No chat history found to delete.")

        elif action == "Provide Feedback":
            feedback_text = st.text_area("Your Feedback:")
            if st.button("Submit Feedback"):
                if len(feedback_text) > 10:
                    store_feedback(st.session_state['sessionid'], feedback_text)
                    st.success("Feedback submitted successfully!")
                else:
                    st.warning("Please write feedback before submitting.")

    elif st.session_state['user_role'] == "admin":
        st.title("Admin Dashboard")
        admin_option = st.sidebar.selectbox("Select an option", ["Upload Document", "View Interactions", "View Feedback", "Create User"])

        if admin_option == "Upload Document":
            st.header("Upload a Document")
            uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "csv", "xlsx", "xls", "doc", "md"])
            if uploaded_file is not None:
                result = upload_document(uploaded_file)
                st.success(result)

        elif admin_option == "View Interactions":
            st.header("Chat Interactions")
            getuserid = st.text_input("Enter UserID:")
            if st.button("Get Data") and getuserid:
                interactions = fetch_chat_interactions(getuserid)
                if interactions :
                    st.subheader(f"User: {interactions[0]['SessionId']}")
                    for interaction in interactions:                    
                        chatuser = json.loads(interaction['History'])
                        if chatuser['type'] == 'human':
                            st.write(f"Query: {chatuser['data']['content']}")
                            #st.write(f"{type(chatuser['data']['content'])}")
                        if chatuser['type'] == 'ai':
                            st.write(f"Response: {(chatuser['data']['content'])}")


        elif admin_option == "View Feedback":
            st.header("User Feedback")
            getuserid = st.text_input("Enter UserID:")
            if st.button("Get Data") and getuserid:
                feedback_list = fetch_feedback(getuserid)
                if feedback_list:
                    st.subheader(f"User: {feedback_list[0]['userid']}")
                    for feedback in feedback_list:                    
                        st.text(f"Feedback: {feedback['feedback']}")

        elif admin_option == "Create User":
            st.header("Create a New User")
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            new_userrole = st.selectbox("Select Role", ["end_user", "admin"])
            if st.button("Create User"):
                result = create_user(new_username, new_password, new_userrole)
                if result == "User created successfully":
                    st.success(result)
                else:
                    st.error(result)
