import os
import sys
import time
import gradio as gr

import yake

import titan_ea_demo.config as config
import titan_ea_demo.utils as utils
import titan_ea_demo.dag_functions as dag

from typing_extensions import TypedDict
from typing import List

from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

topics_in_vector_store = ""

def rag_clear_training():
    global topics_in_vector_store
    topics_in_vector_store = ""
    delete_handle = Chroma(
        client=config.CHROMADB_CLIENT,
        collection_name="rag-chroma"
    )
    delete_handle.delete_collection()
    yield "Cleared past training data\n"
    

def rag_training(urls):
    global topics_in_vector_store
    url_list = utils.url_splitter(urls)
    text_token_size = 500
    text_token_overlap = 0
    docs = []
    output = ""
    start_time = time.time()
    for url in url_list:
        output = f"{output}Fetching {url}\n"
        yield output
        # retrieve all the documents in the corpus URL list
        docs.append(WebBaseLoader(url).load())
    output = f"{output}{utils.time_stamper(start_time)}"
    yield output
    start_time = time.time()
    docs_list = [item for sublist in docs for item in sublist]
    # split the documents text into searchable chunks
    output = f"{output}Tokenizing downloaded content\n"
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=text_token_size, chunk_overlap=text_token_overlap
    )
    doc_splits = text_splitter.split_documents(docs_list)
    output = f"{output}{utils.time_stamper(start_time)}"
    yield output
    start_time = time.time()
    # use GPT4ALL model to create emeddings and add them to a local vector DB
    # LLM
    output = f"{output}Creating vector DB from tokenized content\n"
    yield output
    output = f"{output}Deleting any past vector ids\n"
    yield output
    rag_clear_training()
    output = (
        f"{output}Creating vector DB query agent for top 3 documents by similarity\n"
    )
    yield output
    vectorstore = utils.get_vectorstore(doc_splits)
    output = f"{output}RAG Corpus Training complete with {len(docs)} documents in {len(doc_splits)} tokens"
    output = (
        f"{output} of {text_token_size} bytes with {text_token_overlap} byte overlap.\n"
    )
    output = f"{output}{len(vectorstore)} embedded vectors stored.\n"
    output = f"{output}{utils.time_stamper(start_time)}"
    yield output
    output = f"{output}Creating relevance keyword cache.\n"
    start_time = time.time()
    kw_extractor = yake.KeywordExtractor()
    global_keywords = []
    for doc in docs:
        for index, word in enumerate(
            kw_extractor.extract_keywords(doc[0].page_content)
        ):
            if index < 5 and word[0] not in global_keywords:
                global_keywords.append(word[0])
    output = f"{output} Extracted {len(global_keywords)} keywords from corpus: {global_keywords}\n"
    topics_in_vector_store = ",".join(global_keywords)
    output = f"{output}{utils.time_stamper(start_time)}"
    yield output

### State
class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        output: running output for the chat bot
    """

    question: str = ""
    generation: str = ""
    web_search: bool = False
    documents: List[str] = []
    retry_generation: bool = False
    answered_question: bool = False
    output: str = ""


def rag_inference(question, chat_history):
    state = AgentState(question=question)
    state["output"] = "Let me think...\n"
    yield state["output"]
    state = dag.route_question(state, topics_in_vector_store)
    if state["web_search"]:
        yield state["output"]
        state = dag.web_search(state)
        yield state["output"]
    else:
        yield state["output"]
        state = dag.retrieve(state)
        yield state["output"]
        state = dag.grade_documents(state)
        yield state["output"]
        if state["web_search"]:
            state = dag.web_search(state)
            yield state["output"]
    if 'documents' in state and len(state['documents']) > 0:
        state = dag.generate(state)
        yield state["output"]
        state = dag.grade_generation_v_documents_and_question(state)
        yield state["output"]
        retries = 0
        while state["retry_generation"] and retries < 3: 
            retries += 1
            state = dag.generate(state)
            yield state["output"]
            state = dag.grade_generation_v_documents_and_question(state)
            yield state["output"]
            if state["answered_question"]:
                break
    else:
        state['answered_question'] = False
    if not state["answered_question"]:
            output = state["output"]
            generation = state["generation"]
            yield f"{output}I don't think this answers your question well.. but here is my best guess:\n\n{generation}"


def main():
    
    if os.getenv("TAVILY_API_KEY") is None or os.getenv("TAVILY_API_KEY") == '':
        print(
            f"You need to define the TAVILY_API_KEY environment variable or web search won't work... exiting"
        )
        sys.exit(1)
    
    print(f"Checking if Ollama is serving: {config.LOCAL_LLM}")
    if not utils.ollama_pull_model():
        print(f"Ollama already serving: {config.LOCAL_LLM}")
    else:
        print(f"Ollama {config.LOCAL_LLM} now loaded")


    gr.set_static_paths(paths=["assets/"])
        
    with gr.Blocks() as demo_app:
        with gr.Row():
            with gr.Column(scale=1, min_width=140):
                gr.Image(value='assets/Titan_Badge_128.png', height=160, width=128, show_download_button=False, show_label=False)
            with gr.Column(scale=10, min_width=600):
                gr.Markdown(
                """ 
                # <span style="color:lightblue">Titan AI EA Demo Application</span>
                This demonstration uses a local instance of *llama3* LLM and *GPT4All* embedding models to perform an adaptive RAG with corrective web search and self-corrective grading of the generative responses.         
                """
                )
        with gr.Tab("Ask Me a Question"):
            answer = gr.Textbox(
                label="Answer:"
            )
            msg = gr.Textbox(
                label="Question:"
            )
            clear = gr.ClearButton([msg, answer])
            msg.submit(rag_inference, [msg, answer], [answer])
        with gr.Tab("RAG Corpus Training"):
            urls = gr.Textbox(
                label="URLs to Download Document Corpus",
                lines=10,
                max_lines=30,
                value="\n".join(config.DEFAULT_URLS)
            )
            clear = gr.Button("Clear Corpus")
            submit = gr.Button("Start Training")
            out = gr.Textbox(
                label="Training Console Output"
            )
            submit.click(rag_training, urls, out)
            clear.click(rag_clear_training, None, out)
            
        with gr.Tab("About RAG"):
            with open(config.ABOUT_MARKDOWN_FILE, 'r') as md_file:
                gr.Markdown(md_file.read())
                
    demo_app.css="footer {visibility: hidden}"
    demo_app.launch(share=False, server_name=config.API_LISTENER_ADDRESS, server_port=int(config.API_LISTENER_PORT))


if __name__ == "__main__":
    main()
