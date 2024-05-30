import os
import sys
import time
import gradio as gr
import ollama
import chromadb
import yake

from dotenv import load_dotenv
from datetime import datetime

from typing_extensions import TypedDict
from typing import List

from langchain.schema import Document

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

load_dotenv()

default_urls = [
    "https://clouddocs.f5.com/service-proxy/latest/intro.html",
    "https://clouddocs.f5.com/service-proxy/latest/spk-bgp-overview.html",
    "https://clouddocs.f5.com/service-proxy/latest/spk-network-overview.html",
    "https://clouddocs.f5.com/service-proxy/latest/spk-secure-spk-deployment.html",
    "https://clouddocs.f5.com/service-proxy/latest/spk-tmm-resources.html",
    "https://clouddocs.f5.com/cnfs/robin/latest/intro.html",
    "https://clouddocs.f5.com/cnfs/robin/latest/cnf-firewall-crd.html",
    "https://clouddocs.f5.com/cnfs/robin/latest/cnf-context-global.html",
]

ollama_host = "localhost"
if not os.getenv("CHROMADB_HOST") == "None":
    ollama_host = os.getenv("OLLAMA_HOST")

chromadb_host = "localhost"
if not os.getenv("CHROMADB_HOST") == "None":
    chromadb_host = os.getenv("CHROMADB_HOST")

web_search_tool = TavilySearchResults(k=20)

vector_retriever = None
topics_in_vector_store = ""


def time_stamper(start_time):
    run_duration = round((time.time() - start_time), 3)
    return f"Runtime: ({run_duration} seconds)\n"


def clear_training(urls):
    global vector_retriever, topics_in_vector_store
    topics_in_vector_store = ""
    if chromadb_host:
        client = chromadb.HttpClient(host=chromadb_host)
        yield "Cleared"
        delete_handle = Chroma(
            client=client,
            collection_name="rag-chroma"
        )
        vector_retriever = None


def training_requests(urls):
    global vector_retriever, topics_in_vector_store
    url_list = urls.split("\n")
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
    output = f"{output}{time_stamper(start_time)}"
    yield output
    start_time = time.time()
    docs_list = [item for sublist in docs for item in sublist]
    # split the documents text into searchable chunks
    output = f"{output}Tokenizing downloaded content\n"
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=text_token_size, chunk_overlap=text_token_overlap
    )
    doc_splits = text_splitter.split_documents(docs_list)
    output = f"{output}{time_stamper(start_time)}"
    yield output
    start_time = time.time()
    # use GPT4ALL model to create emeddings and add them to a local vector DB
    # LLM
    output = f"{output}Creating vector DB from tokenized content\n"
    yield output
    vectorstore = None
    if chromadb_host:
        client = chromadb.HttpClient(host=chromadb_host)
        output = f"{output}Deleting any past vector ids"
        yield output
        delete_handle = Chroma(
            client=client,
            collection_name="rag-chroma"
        )
        delete_handle.delete_collection()
        vectorstore = Chroma.from_documents(
            client=client,
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf"),
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf"),
        )
    # create a query retriever for the new vector store
    # let's limit to 3 document to return based on similarity
    output = (
        f"{output}Creating vector DB query agent for top 3 documents by similarity\n"
    )
    yield output
    
    
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    output = f"{output}RAG Corpus Training complete with {len(docs)} documents in {len(doc_splits)} tokens"
    output = (
        f"{output} of {text_token_size} bytes with {text_token_overlap} byte overlap.\n"
    )
    output = f"{output}{len(vectorstore)} embedded vectors stored.\n"
    output = f"{output}{time_stamper(start_time)}"
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
    output = f"{output}{time_stamper(start_time)}"
    yield output


# Inference items
local_llm = "llama3"

from langchain_community.chat_models import ChatOllama

# LLM
text_llm = None
json_llm = None
if ollama_host:
    text_llm = ChatOllama(
        model=local_llm, temperature=0, base_url=f"http://{ollama_host}:11434/"
    )
    json_llm = ChatOllama(
        model=local_llm,
        format="json",
        temperature=0,
        base_url=f"http://{ollama_host}:11434/",
    )
else:
    text_llm = ChatOllama(model=local_llm, temperature=0)
    json_llm = ChatOllama(model=local_llm, format="json", temperature=0)


doc_relevance_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)
retrieval_grader = doc_relevance_prompt | json_llm | JsonOutputParser()

generate_concise_answer_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Provide your answer in three sentences maximum and keep the answer concise. Answer the question so a 5th grade can understand, and
    do not mention the provided content.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)
rag_chain = generate_concise_answer_prompt | text_llm | StrOutputParser()

hallucination_prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)
hallucination_grader = hallucination_prompt | json_llm | JsonOutputParser()

answer_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)
answer_grader = answer_grader_prompt | json_llm | JsonOutputParser()

source_router_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on {topics_in_vector_store}. 
    You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search. 
    Give a binary choice 'web_search' or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)
question_router = source_router_prompt | json_llm | JsonOutputParser()


### State
class GraphState(TypedDict):
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


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    
    question = state["question"]
    # Retrieval
    documents = vector_retriever.invoke(question)
    web_search = state["web_search"]
    output = state["output"]
    output = f"{output}I've read something about this.\nI got documents with similary information.\nGive me a chance to skim them and make sure I answer correctly.\n"
    return {"documents": documents, "web_search": web_search, "question": question, "output": output}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """
    
    question = state["question"]
    documents = []

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    web_search = state["web_search"]
    output = state["output"]
    output = f"{output}Looks like they found something for you.\nLet me read through what they sent me to get you an answer.\n"
    return {"documents": documents, "web_search": web_search, "question": question, "output": output}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    web_search = state["web_search"]
    output = state["output"]
    output = f"{output}I have an answer, but I like to double check I'm not making things up.\n"
    return {"documents": documents, "question": question, "web_search": web_search, "generation": generation, "output": output}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    question = state["question"]
    documents = state["documents"]
    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            filtered_docs.append(d)
        # Document not relevant
        else:
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            continue
    output = state["output"]
    web_search = False
    if len(filtered_docs) == 0:
        web_search = True
        output = f"{output}These document don't seem to answer your question.\nLet me go back to the web AIs for you.\n"
    else:
        output = f"{output}I got some good docs here.\n"    
    return {"documents": filtered_docs, "question": question, "web_search": web_search, "output": output}



### Conditional edge


def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state adding new attribute for the source of the documents

    Returns:
        str: Next node to call
    """
    question = state["question"]
    output = state["output"]
    source = question_router.invoke(
        {"question": question, "topics_in_vector_store": topics_in_vector_store}
    )
    # Nobody ran RAG training
    if vector_retriever is None:
        output = f"{output}I don't have any local documents to read.\nLet me ask my big brother web AIs about it for you.\n"
        return {"web_search": True, "question": question, "output": output}
    elif source["datasource"] == "web_search":
        output = f"{output}Doesn't look like I know much about your question.\nLet me ask my big brother web AIs about it for you.\n"
        return {"web_search": True, "question": question, "output": output}
    elif source["datasource"] == "vectorstore":
        output = f"{output}Glad you asked me!\n"
        return {"web_search": False, "question": question, "output": output}


### Conditional edge


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    web_search = state["web_search"]
    output = state["output"]
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = False
    if "score" in score and score["score"] == "yes":
        grade = True
    if "grounded" in score and score["grounded"]:
        grade = True
    if "Grounded" in score and score["Grounded"]:
        grade = True

    # Check hallucination
    if grade:
        output = f"{output}Yeah.. I'm happy enough that I'm not making things up. Just double checking my answer.\n"
        # Check question-answering
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            output = f"{output}\n\n{generation}\n\n"
            return {"documents": documents, "question": question, "web_search": web_search, "generation":generation, "retry_generation": False, "answered_question": True, "output": output}
        else:
            output = f"{output}I'm not happy with my answer.. Let me try again.\n"
            return {"documents": documents, "question": question, "web_search": web_search, "generation":generation, "retry_generation": True, "answered_question": False, "output": output}
    else:
        output = f"{output}Nope.. I think I'm talking off the top of my head.. just making things up. Let me try again.\n"
        return {"documents": documents, "question": question, "web_search": web_search, "generation":generation, "retry_generation": True, "answered_question": False, "output": output}


def inference_requests(question, chat_history):
    state = GraphState(question=question)
    state["output"] = "Let me think...\n"
    yield state["output"]
    state = route_question(state)
    if state["web_search"]:
        yield state["output"]
        state = web_search(state)
        yield state["output"]
    else:
        yield state["output"]
        state = retrieve(state)
        yield state["output"]
        state = grade_documents(state)
        yield state["output"]
        if state["web_search"]:
            state = web_search(state)
            yield state["output"]
    if 'documents' in state and len(state['documents']) > 0:
        state = generate(state)
        yield state["output"]
        state = grade_generation_v_documents_and_question(state)
        yield state["output"]
        retries = 0
        while state["retry_generation"] and retries < 3: 
            retries += 1
            state = generate(state)
            yield state["output"]
            state = grade_generation_v_documents_and_question(state)
            yield state["output"]
            if state["answered_question"]:
                break
    else:
        state['answered_question'] = False
    if not state["answered_question"]:
            output = state["output"]
            generation = state["generation"]
            yield f"{output}I don't think this answers your question well.. but here is my best guess:\n\n{generation}"


def ollama_pull_model():
    ollama_client = ollama.Client(host=f"http://{ollama_host}:11434")
    loaded_modules = ollama_client.list()
    llama_needed = True
    for model in loaded_modules['models']:
        if local_llm in model['name']:
            return False
    ollama_client.pull(local_llm)
    return True


def main():
    print(f"Connecting to Ollama at: {ollama_host}")
    print(f"Connecting to ChromaDB at: {chromadb_host}")
    
    print(f"Checking if Ollama is serving: {local_llm}")
    if not ollama_pull_model():
        print(f"Ollama already serving: {local_llm}")
    else:
        print(f"Ollama {local_llm} now loaded")

    if os.getenv("TAVILY_API_KEY") is None or os.getenv("TAVILY_API_KEY") == '':
        print(
            f"You need to define the TAVILY_API_KEY environment variable or web search won't work... exiting"
        )
        sys.exit(1)

    gr.set_static_paths(paths=["assets/"])

    training_interface = gr.Interface(
        fn=training_requests,
        inputs=[
            gr.Textbox(
                label="URLs to Download Document Corpus",
                lines=10,
                max_lines=30,
                value="\n".join(default_urls),
            )
        ],
        outputs=[gr.Textbox(label="Output Console")],
    )

    chat_interface = gr.ChatInterface(
        fn=inference_requests,
        examples=["What features does SPK support?"],
        title="Tiny Tim - Your Local AI",
        fill_height=True,
        css="component-17 {height: 600px;}"
    )
    
    tabbed_interface = gr.TabbedInterface(
            [chat_interface, training_interface],
            ["Ask Me a Question", "RAG Corpus Training"],
            css="footer {visibility: hidden}"
    )
    
    #demo_app = tabbed_interface
    
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
            msg.submit(inference_requests, [msg, answer], [answer])
        with gr.Tab("RAG Corpus Training"):
            urls = gr.Textbox(
                label="URLs to Download Document Corpus",
                lines=10,
                max_lines=30,
                value="\n".join(default_urls)
            )
            clear = gr.Button("Clear Corpus")
            submit = gr.Button("Start Training")
            out = gr.Textbox(
                label="Training Console Output"
            )
            submit.click(training_requests, urls, out)
            clear.click(clear_training, urls, out)
            
        with gr.Tab("About RAG"):
            gr.Markdown(
                """
                # <span style="color:lightblue">R.A.G. (Retrieval Augmented Generation)</span>
                LLM are trained on vast amounts of language examples to be able to produce a predictive natural language outcome to prompted inputs.

                However, don't ask an LLM to produce a natural language response based on any proprietary language or vocabulary which was not in their training data.

                How can we take advantage of the natural language generation capabilities of LLMs, but augment their training data with a corpus of our own proprietary knowledge? That is where R.A.G. comes in. 

                ## <span style="color:lightblue">Training in R.A.G.</span>

                LLM processing requirements and speed of response are highly dependent on the complexity of the prompted request made of them. While you could ask the LLM to generate a response based on the whole of your corpus of data, not only would that not yield the response accuracy you are looking for, but it would also heat the planet for extended periods of time. All around a bad idea.

                So how do we narrow down a corpus of information to just related language? We have been using search engines to do that for years. First, we chop down our content into meaningful, but manageable chucks, or tokens, which can be processed quickly by parallel processes, like the ones GPU stream processor are so good at doing quickly. We can do both indexed keywords searching, which is good at narrowing down our content to specific words, as well as semantic search based on numerical analysis of the language in our corpus. Keyword extraction is pretty simply. To do the semantic search we need to turn our tokenized text into numerical embedded vectors which can be stored and searched in a vector database. We can then search the vector database for all the text that is highly similar semantically to the prompt question being asked.

                Combining our LLM which generates natural language response with a searchable pile of corpus data to base a response we can create an inferencing workflows which will give us what we want.

                ## <span style="color:lightblue">Inferencing in R.A.G.</span>

                In this apps corrective R.A.G. workflow, when a question is received, we can first use our LLM to ask if the question being asked is related to our pile of keywords. That will give us a quick and dirty way to decide if the question is related to our corpus at all. If the question is not related, we just do a web search for documents and get a temporary set of documents to use to generate the response.

                If we determine that the question is related to our corpus, we quickly use the same embedding model which turned our corpus tokens into numerical vectors and then query our vector database to return only a few documents related to our question which we will in our prompt to get the LLM to generate a natural language response. We tell the LLM to answer the question based on this highly reduced set of document tokens the vector database told us were related to our question.

                But what about those reports of LLM predicting language that just 'looses its mind' and makes no sense to our context. Those are called LLM hallucinations. We will put two corrective measures in place in our workflow to assist with that problem. 

                First, we will take another pass at our LLM to assure that the generated response is 'grounded' in the corpus of the relevant text we supplied. We do that with some fancy 'prompt engineering'. If the generated is not 'grounded', then the LLM lost its mind. That is the first thing we do.. get grounded. (I was always grounded as a kid, so I'm used to it!)

                Second, we will ask the LLM if the answer it supplied actually answers the question. Again, just some fancy prompt work. This is a summarization type task that LLMs are really good at doing. 

                Once we:
                   -	Have a set of relevant document tokens from our corpus or the web
                   -	Asked the LLM to answer the question from that set of relevant document tokens
                   -	Asked the LLM if the generated response is 'grounded' in the document tokens
                   -	Asked the LLM if the generated response answered the question

                We can confidently return a generated response to the prompted question. If not, we can tell the LLM to generate another response a few times to see if it can do any better. Remember, the generated response has some creativity and randomness built into the LLM, just like our real language does. 

                So we:

                [x] **Retrieved** a small relevant set of document tokens related to our question. In our case we use a corpus of documents tokenized, keyword indexed, and semantically searched, or else a web search.

                [x] **Augmented**  the LLM data by telling it to answer from the retrieved relevant data.

                [x] **Generated**  an answer with the LLM that is validated from our relevant data and actually answers the prompted question.

                There you have it -> **R.A.G.**

                *Titan AI PM Team*

                """
            )
                
    demo_app.css="footer {visibility: hidden}"
    demo_app.launch(share=False, server_name='0.0.0.0')


if __name__ == "__main__":
    main()
