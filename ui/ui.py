import os
import ollama
import streamlit as st
import tempfile
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import Ollama
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")

def translate_url_to_ssh(url):
    tmp_url = url
    if tmp_url.startswith("https://"):
        tmp_url = url.replace("https://","git@").replace(".com/",".com:")
    if not tmp_url.endswith(".git"):
        tmp_url += ".git"
    tmp_url = tmp_url
    return tmp_url

def list_huggingface_local_models():
    home = f"{os.path.expanduser('~')}/.cache/huggingface/hub"
    non_empty_folders = []
    for folder in os.listdir(home):
        folder_path = os.path.join(home, folder)
        if os.path.isdir(folder_path) and os.listdir(folder_path) and "models" in folder_path:
            format_model = folder.replace("models--","").replace("--","/")
            non_empty_folders.append(format_model)
    return non_empty_folders


prompt_template= """
### [INST] 
Here is the context to help you answer the question do not mention that you had access to this context:

{context}

### QUESTION:
{question} 

[/INST]
"""

def askQuery(query, collection,embedding_model,inference_model):

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    client = QdrantClient(url=QDRANT_URL)
    db = Qdrant(client,collection,embeddings)
    retriever = db.as_retriever()

    # TODO: Handle model load/API Key error
    if inference_model == "OpenAI":
        model = OpenAI()
    else:
        model = Ollama(model=inference_model)

    if embedding_model == "OpenAI":
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    prompt = ChatPromptTemplate.from_template(prompt_template)

    qa = RetrievalQA.from_chain_type(
        llm=model, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    response = qa.invoke(query)
    return(response)

def collection():
    import git
    import os
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    from langchain_community.vectorstores import Qdrant
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    st.write('# Github Markdown Collection Creation 📚')

    repo_url = st.text_input("repo url",placeholder="https://github.com/yourproject/yourrepo")
    #TODO: Add a check for valid git url
    update= st.button("Create/Update Collection",disabled=repo_url == "")
    if update:
        with st.status("RAG creation in progress..."):
            with tempfile.TemporaryDirectory() as tmpdirname:
                st.write("Cloning the repo 📥")
                repo_name = repo_url.split("/")[-1].replace(".git","")
                repo = git.Repo.clone_from(translate_url_to_ssh(repo_url),f'{tmpdirname}/{repo_name}')
                st.write('Parsing the markdown files 📄')
                files_in_folder = os.listdir(f'{tmpdirname}/{repo_name}')
                md_files = [file for file in files_in_folder if file.endswith('.md')]

                docs = []

                headers_to_split_on = [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]

                st.write('Embedding 📄🔗')
                for md in md_files:
                    with open(f'{tmpdirname}/{repo_name}/{md}', 'r', encoding='utf-8') as file:

                        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                        docs.extend(markdown_splitter.split_text(file.read()))
                
                st.write('Saving the collection 📚')
                if collection.embedding_model == "OpenAI":
                    embeddings = OpenAIEmbeddings()
                else:
                    embeddings = HuggingFaceEmbeddings(model_name=collection.embedding_model)
                
                db = Qdrant.from_documents(
                    docs,
                    embeddings,
                    url=QDRANT_URL,
                    collection_name=repo_name,
                    optimizers_config=None
                )

def chat():
    st.write("# RAG ✨")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Ask something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            response = askQuery(prompt, _vector_db_collection,_embedding_model,_llm_model)
            st.session_state.messages.append({"role": "assistant", "content": response['result']})
            st.write(response['result'])


index = {
    "Create/Update collection": collection,
    "Chat": chat,
}

embedding_models = list_huggingface_local_models()
embedding_models.append("OpenAI")
llm_models = [model['name'] for model in ollama.list()["models"]]
llm_models.append("OpenAI")
Qclient = QdrantClient(url=QDRANT_URL)
collections = [collection.name for collection in Qclient.get_collections().collections]
Qclient.close()



_demo_name = st.sidebar.selectbox("RAG", index.keys())

st.sidebar.write("## VectorDB")
_vector_db_collection = st.sidebar.selectbox("Available Collections", collections)
st.sidebar.write("## Embedding")
_embedding_model = st.sidebar.selectbox("Model", embedding_models)
_semantichunk = st.sidebar.checkbox("Use Semantic Chunker",disabled=True)
st.sidebar.write("## Inference Model")
_llm_model = st.sidebar.selectbox("LLM", llm_models)

if _llm_model == "OpenAI" or _embedding_model == "OpenAI":
    st.sidebar.write("## Credentials")
    _openai_api_key = st.sidebar.text_input("OpenAI API Key")
st.sidebar.write("## Advanced")
st.sidebar.markdown(f'[Open Qdrant Dashboard]({QDRANT_URL}/dashboard)', unsafe_allow_html=True)
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    st.sidebar.markdown(f'[Open langsmith Dashboard](https://smith.langchain.com)', unsafe_allow_html=True)


index[_demo_name]()