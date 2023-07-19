import os
from pathlib import Path
from llama_index import (
    download_loader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from dotenv import load_dotenv


JSONReader = download_loader("JSONReader")

loader = JSONReader()
documents = loader.load_data(Path("./scraped_data.json"))
load_dotenv()
index = VectorStoreIndex.from_documents(documents)

index.storage_context.persist()
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")
# load index
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
response = query_engine.query("how do i rollback an application? provide a link")
print(response)
