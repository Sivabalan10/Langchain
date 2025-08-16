import os
import pickle
import time
import langchain
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
import datetime 


from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model = "llama-3.1-8b-instant",
    max_retries=2,
)

urls = []

st.title("Research Tool - GENAI")

st.sidebar.title("Add your source")



for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

filename_vector_db = st.sidebar.text_input("Enter your space name")
process_url_clicked = st.sidebar.button("Process URL")

main_place_holder = st.empty()

if process_url_clicked:
    try:
        # Loader to load the data
        loader = UnstructuredURLLoader(urls=urls)
        main_place_holder.text("Data Loading Started .... ⏳⏳⏳")

        data = loader.load()

        # Splitting the data 
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n','\n','.',','],
            chunk_size=600
        )
        main_place_holder.text("Text split started.....⏳⏳⏳")
        docs = text_splitter.split_documents(data)

        # create embeddings

        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorindex = FAISS.from_documents(docs, embedding_function)
        main_place_holder.text('Embedding vector started building....🪓🪓🪓')

        time.sleep(2)
        # Storing vector index create in local
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path=f"vector_db_faiss/{filename_vector_db}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(vectorindex, f)
            main_place_holder.text("Successfully stored the sources ✅")
            time.sleep(2)
            log_m = f"Space created in the name of {filename_vector_db}"
            main_place_holder.text(log_m)
            time.sleep(2)
            main_place_holder.text("")

    except Exception as e:
        main_place_holder.text(e)
        time.sleep(2)
        main_place_holder.text("Something went wrong. Please give valid URLs")
        time.sleep(2)
        main_place_holder.text("")
    

# List available vector DB files
vector_db_folder = "vector_db_faiss"
available_files = [
    f[:-4] for f in os.listdir(vector_db_folder) if f.endswith(".pkl")
]

selected_space = st.sidebar.selectbox(
    "Select your space", available_files
)

selected_file_path = os.path.join(vector_db_folder, f"{selected_space}.pkl")
    
query = main_place_holder.text_input("Questions:")
if query:
    if os.path.exists(selected_file_path):
        print(selected_file_path)
        with open(selected_file_path, "rb") as f:
            vectorstore = pickle.load(f) 

        # constructing the chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        st.subheader(result["answer"])
        
        # show the sources
        sources = result.get("sources","")
        if sources:
            st.subheader("Sources:")
            source_list = sources.split('\n')
            for source in source_list:
                st.write(source)



        
