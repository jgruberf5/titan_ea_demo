FROM python:3.11-slim
WORKDIR /code
COPY ./ /code
RUN pip install -r requirements.txt

ENV OLLAMA_HOST='localhost'
ENV CHROMADB_HOST='localhost'
ENV LANGCHAIN_TRACING_V2='false'
ENV LANGCHAIN_ENDPOINT='https://api.smith.langchain.com'
ENV LANGCHAIN_API_KEY=''
ENV TAVILY_API_KEY=''

EXPOSE 7860

ENTRYPOINT ["python3", "-m", "titan_ea_demo"]