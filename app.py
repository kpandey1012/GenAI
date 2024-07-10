from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import CSVLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader,PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader,UnstructuredHTMLLoader
import os
import time
import textwrap



app = Flask(__name__)

folder_path = "db"


from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

chat_message_history = MongoDBChatMessageHistory(
    session_id="test_session",
    connection_string="mongodb://localhost:27017",
    database_name="my_db",
    collection_name="chat_histories",
)

local_llm = ChatOllama(model="phi3")

embedding = OllamaEmbeddings(model='mxbai-embed-large')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=200, length_function=len, is_separator_regex=False
)

DOCUMENT_MAPPING = {
    ".html": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}


@app.route("/general_quetions", methods=["POST"])
def general_quetions():    
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = local_llm.invoke(query)

    print(response)

    response_answer = {"answer": response}
    return response_answer


@app.route("/user_questions", methods=["POST"])
def llmm_response():    
    json_content = request.json
    query = json_content.get("query")
    userid = json_content.get("userid")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    print("Creating QA chain")
    ### Contextualize question ###
    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )
    history_aware_retriever = create_history_aware_retriever(
    local_llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )
    question_answer_chain = create_stuff_documents_chain(local_llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string="mongodb://localhost:27017",
        database_name="my_db",
        collection_name="chat_histories",
    ),
    input_messages_key="input",
    history_messages_key="chat_history",
    )
    config = {"configurable": {"session_id": userid}}
    
    print("getting response from llm")

    result = chain_with_history.invoke({"input": query}, config=config)
    
    return  result


@app.route("/upload_docs", methods=["POST"])
def upload_documents():
    file = request.files["file"]
    file_name = file.filename
    save_file = "files/" + file_name
    file.save(save_file)
    file_extension = os.path.splitext(save_file)[1]
    loader_class = DOCUMENT_MAPPING.get(file_extension)
    if loader_class:
        loader = loader_class(save_file)
    
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()
    os.remove(save_file)
    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()