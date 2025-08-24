import os
import warnings
import logging

import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from groq import Groq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title ("ANKITA PATEL's CHATBOT!")
#setup a session state variable to hold all the old messages
if'messages' not in st.session_state:
    st.session_state.messages=[]
#display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])
uploaded_file=st.file_uploader("Upload a pdf",type=["pdf"])
if uploaded_file is not None:
    with open("temp.pdf","wb") as f:
        f.write(uploaded_file.read())
    pdf_path="temp.pdf"
else:
    pdf_path="reflexion.pdf"

if uploaded_file is None:
    st.warning("Please upload a pdf to start chatting othe")

@st.cache_resource
def get_vectorstore(pdf_path):
    # uploaded_file=st.file_uploader("Upload a pdf",type=["pdf"])
    # if uploaded_file is not None:
    #     with open("temp.pdf","wb") as f:
    #         f.write(uploaded_file.read())
    #     pdf_path="temp.pdf"
    # else:
    #     pdf_path=None

    loaders=[PyPDFLoader(pdf_path)]
    #create chunks,aka vectors(chromedb)
    index=VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore


st.caption(f"Currently answering from:{pdf_path}")

prompt=st.chat_input("Pass your prmopt here!")


if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role':'user','content':prompt})
    
    groq_sys_prompt=ChatPromptTemplate.from_template("""You are my sweetest partner to answer my queries!{user_prompt}""")
    model="Llama3-8b-8192"
    groq_chat=ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model
    )

    try:
        vectorstore=get_vectorstore(pdf_path)
        if vectorstore is None:
            st.error("Failed to load the document")

        else: 
            chain=RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs=({"k":3})),
                return_source_documents=True
            )
            
            result=chain({"query":prompt})
            response=result["result"]
    
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append(
                {'role':'assistant','content':response})
    except Exception as e:
        st.error(f"Error:[{str(e)}]")