# RAG for Github

## Requirements
* Ollama
* Docker

## Instructions

1) Install python requirements
   ```
   pip intall -r requirements.txt
   ```

2) Install Ollama following official installation instructions : [https://github.com/ollama/ollama](https://github.com/ollama/ollama)

3) Pull your required local LLM
   ```
   ollama pull llama2
   ```

4) Copy/Edit `.env.template` as `api/.env` and `ui/.env` file with your OpenAI and LangSmith API key. [https://smith.langchain.com](https://smith.langchain.com)


4) Run Qdrant 
   ```
   docker run -dp 6333:6333 qdrant/qdrant
   ```


## UI

1) Start streamlit ui
   ```
   streamlit run ui/ui.py
   ```
2) Open Web browser to `http://localhost:8501`

3) 


## API

1) Start API
   ```
   python api/api.py
   ```

2) Open Web browser to `http://localhost:5000/docs`