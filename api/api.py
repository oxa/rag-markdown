import git
import os
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from qdrant_client import QdrantClient, AsyncQdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.llms import Ollama
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

app = FastAPI()

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")

class Collection(BaseModel):
    git_url: str
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"

class Query(BaseModel):
    query: str
    collection: str
    inference_model: str = "llama2"
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"


def translate_url_to_ssh(url):
    tmp_url = url
    if tmp_url.startswith("https://"):
        tmp_url = url.replace("https://","git@").replace(".com/",".com:")
    if not tmp_url.endswith(".git"):
        tmp_url += ".git"
    tmp_url = tmp_url
    return tmp_url

prompt_template= """
### [INST] 
Here is the context to help you answer the question do not mention that you had access to this context:

{context}

### QUESTION:
{question} 

[/INST]
"""

async def askQuery(query, collection,embedding_model,inference_model):

    # TODO: Handle model load/API Key error
    if inference_model == "OpenAI":
        model = OpenAI()
    else:
        model = Ollama(model=inference_model)

    if embedding_model == "OpenAI":
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    aclient = AsyncQdrantClient(url=QDRANT_URL)
    client = QdrantClient(url=QDRANT_URL)

    db = Qdrant(client,async_client=aclient, collection_name=collection,embeddings=embeddings)
    retriever = db.as_retriever()



    # Abstraction of Prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)

    qa = RetrievalQA.from_chain_type(
        llm=model, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    response = await qa.ainvoke(query)
    return(response)

@app.post("/collections")
async def create_collection(collection: Collection):

    with tempfile.TemporaryDirectory() as tmpdirname:
        repo_name = collection.git_url.split("/")[-1].replace(".git","")
        repo = git.Repo.clone_from(translate_url_to_ssh(collection.git_url),f'{tmpdirname}/{repo_name}')
        
        files_in_folder = os.listdir(f'{tmpdirname}/{repo_name}')
        md_files = [file for file in files_in_folder if file.endswith('.md')]

        docs = []

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        for md in md_files:
            with open(f'{tmpdirname}/{repo_name}/{md}', 'r', encoding='utf-8') as file:

                markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                docs.extend(markdown_splitter.split_text(file.read()))
        
        if collection.embedding_model == "OpenAI":
            embeddings = OpenAIEmbeddings()
        else:
            embeddings = HuggingFaceEmbeddings(model_name=collection.embedding_model)
        
        db = Qdrant.from_documents(
            docs,
            embeddings,
            url=QDRANT_URL,
            collection_name=repo_name,
        )
    return {"message": "Collection created successfully", "collection_name": repo_name}

@app.get("/collections")
async def get_collections():
    client = AsyncQdrantClient(url=QDRANT_URL)
    collections = [collection.name for collection in (await client.get_collections()).collections]
    return {"collections": collections}

@app.post("/chat")
async def query_llm(query: Query):
    response = await askQuery(query.query, query.collection,query.embedding_model,query.inference_model)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=5000, debug=True)