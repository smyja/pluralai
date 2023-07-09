import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from llama_index import download_loader, VectorStoreIndex, StorageContext, load_index_from_storage
from dotenv import load_dotenv
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
def query_data(request: QueryRequest):
    JSONReader = download_loader("JSONReader")
    loader = JSONReader()
    documents = loader.load_data(Path('./scraped_data.json'))
    index = VectorStoreIndex.from_documents(documents)

    index.storage_context.persist()
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(request.question)

    if not response:
        raise HTTPException(status_code=404, detail="No results found")

    return QueryResponse(answer=str(response))
