import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE-API-KEY")
PINECONE_ENV="us-east-1"
PINECONE_INDEX_NAME="medical-index"


UPLOAD_DIR="./uploaded_docs"
os.makedirs(UPLOAD_DIR,exist_ok=True)

#initialize pinecone instance
pc=Pinecone(api_key=PINECONE_API_KEY)
spec=ServerlessSpec(cloud="aws",region=PINECONE_ENV)

existing_indexes=[name for name in pc.list_indexes().names()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index=pc.Index(PINECONE_INDEX_NAME)

#load,split,embed and upsert pdf docs content

def load_vectorstore(uploaded_files):
    embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)
    file_paths=[]

    #1.upload
    for file in uploaded_files:
        save_path=Path(UPLOAD_DIR)/file.filename
        with open(save_path,"wb") as f :
            f.write(file.file.read())
        file_paths.append(str(save_path))

    #2.split(chunking)
    for file_path in file_paths:
        loader=PyPDFLoader(file_path)
        documents=loader.load()

        splitter=RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks=splitter.split_documents(documents)

        texts=[chunk.page_content for chunk in chunks]
        metadata=[chunk.metadata for chunk in chunks]
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]


        #3.embedding
        print(f"Embedding chunks")
        embedding=embedding_model.embed_documents(texts)

        #4.apsert to database
        print("Upserting Embeddings...")
        with tqdm(total=len(embedding),desc="Upserting to pinecone") as progress:
            index.upsert(vectors=zip(ids,embedding,metadata))
            progress.update(len(embedding))
        print(f"Upload complete for {file_path}")
