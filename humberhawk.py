import streamlit as st
import os
import base64
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import chromadb
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from streamlit_chat import message
from PIL import Image

st.set_page_config(layout="wide")

CHECKPOINT = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(
    CHECKPOINT, device_map="auto", torch_dtype=torch.float32, offload_folder="save_folder"
)

@st.cache_resource
def data_ingestion():
    file_path = "ilovepdf_merged.pdf"
    loader = PDFMinerLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = chromadb.EphemeralClient()
    # Save texts to Chroma here (assuming you'll update Chroma with the processed texts)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=400,
        do_sample=True,
        temperature=0.4,
        top_p=0.95
    )
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = chromadb.EphemeralClient()
    # You might need to update your retriever construction based on the new Chroma client
    retriever = db.as_retriever()  # Update this accordingly
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=retriever,
    return_source_documents=True
    )

    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="general",
    #     retriever=retriever,
    #     return_source_documents=True
    # )
    return qa


def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa(instruction)
    return generated_text['result']

def main():
    image = Image.open('humberlogo.png')
    st.image(image, use_column_width=True)
    st.title("Humber Hawk")
    
    # Directly ingest and process the specified PDF file
    with st.spinner('Embeddings are in process...'):
        data_ingestion()
    st.success('Embeddings are created successfully!')

    # Get user query and generate a response
    user_input = st.text_input("Ask me a question:", key="input")
    
    if user_input:
        answer = process_answer({'query': user_input})
        st.write("Response:", answer)

if __name__ == "__main__":
    main()
