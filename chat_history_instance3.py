import streamlit as st
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

import os

from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient
client=MongoClient(os.getenv("MONGO_URI"))
db=client["chat_db"]
collection=db["history"]


os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG")
st.write("Upload PDFs and chat ")

api_key = st.text_input("Enter your Groq API key:", type="password")

def inspect(state):
    print("\n--- CURRENT CHAIN STATE ---")
    print(state)
    print("---------------------------\n")
    return state


user_input=""
on = st.toggle("Save chat history feature")
if api_key and not on:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    session_id = "default_session"
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temppdf)
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # --- LCEL REORGANIZATION START ---

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "formulate a standalone question. Do NOT answer it."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

        qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use the context below to answer. "
            "If you don't know, say you don't know. Max 3 sentences.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: contextualize_chain | retriever | format_docs
            )
            | RunnableLambda(inspect)      
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        # --- LCEL REORGANIZATION END ---

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]


        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response_text = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            st.write("Assistant:", response_text)
          
            with st.expander("View Message History"):
                for msg in session_history.messages:
                    st.write(f"{msg.type}: {msg.content}")

elif api_key and on:
    def session_exist(session_id):
        user = collection.find_one({"session_id": session_id})

        if user:
            return True
        return False
    
    def save_message(session_id, role, content):
        collection.update_one(
            {"session_id": session_id},
            {"$push": {"messages": {"role": role, "content": content}}},
            upsert=True
        )

   
    def get_history(session_id):
        data = collection.find_one({"session_id": session_id})
        history = ChatMessageHistory()
        if data:
            for msg in data["messages"]:
                if msg["role"] == "user":
                    history.add_message(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    history.add_message(AIMessage(content=msg["content"]))
        return history
    
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
    
    session_id = st.text_input("Session ID")
    if not session_id:
        st.info("Session ID cannot be empty ")
    if not session_exist(session_id=session_id):
        st.info("No Session exist ")
        st.button("Create new session")
    else:
        st.info("Session found")
       

    print("Session ID:", session_id)

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temppdf)
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "formulate a standalone question. Do NOT answer it."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

        qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use the context below to answer. "
            "Answer ques in 50-60 word and if necessary then explain more"
            "If you don't know, say you don't know. Max 3 sentences.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: contextualize_chain | retriever | format_docs
            )
            | RunnableLambda(inspect)    
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        raw_data = collection.find_one({"session_id": session_id})
        with st.expander("View Message History"):
            if raw_data and "messages" in raw_data:
                for msg in raw_data["messages"]:
                    st.write(f"{msg['role']} : {msg['content']}")
        

        user_input = st.text_input("Your question:")
        if user_input:
            save_message(session_id, "user", user_input)
            session_history = get_history(session_id)
            response_text = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            st.write("Assistant:", response_text)
            save_message(session_id, "assistant", response_text)
            # history = get_history(session_id)
           

else:
    st.warning("Please enter the Groq API Key")


