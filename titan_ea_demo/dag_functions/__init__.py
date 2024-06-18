from titan_ea_demo import chains
from titan_ea_demo import utils

from langchain_core.documents import Document


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
    documents = utils.get_vectorstore_retriever().invoke(question)
    web_search = state["web_search"]
    output = state["output"]
    output = f"{output}I've read something about this.\nI got documents with similary information.\nGive me a chance to skim them and make sure I answer correctly.\n"
    return {
        "documents": documents,
        "web_search": web_search,
        "question": question,
        "output": output,
    }


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
    docs = utils.web_search(question)
    web_results = "\n\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    web_search = state["web_search"]
    output = state["output"]
    output = f"{output}Looks like they found something for you.\nLet me read through what they sent me to get you an answer.\n"
    return {
        "documents": documents,
        "web_search": web_search,
        "question": question,
        "output": output,
    }


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
    generation = chains.rag_chain.invoke({"context": documents, "question": question})
    web_search = state["web_search"]
    output = state["output"]
    output = f"{output}I have an answer, but I like to double check I'm not making things up.\n"
    return {
        "documents": documents,
        "question": question,
        "web_search": web_search,
        "generation": generation,
        "output": output,
    }


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
        score = chains.retrieval_grader.invoke(
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
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
        "output": output,
    }


### Conditional edge


def route_question(state, topics_in_vector_store):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state adding new attribute for the source of the documents

    Returns:
        str: Next node to call
    """
    question = state["question"]
    output = state["output"]
    source = chains.question_router.invoke(
        {"question": question, "topics_in_vector_store": topics_in_vector_store}
    )
    # Nobody ran RAG training
    if utils.get_vectorstore_retriever() is None:
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
    score = chains.hallucination_grader.invoke(
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
        score = chains.answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        grade = score["score"]
        if grade == "yes":
            output = f"{output}\n\n{generation}\n\n"
            return {
                "documents": documents,
                "question": question,
                "web_search": web_search,
                "generation": generation,
                "retry_generation": False,
                "answered_question": True,
                "output": output,
            }
        else:
            output = f"{output}I'm not happy with my answer.. Let me try again.\n"
            return {
                "documents": documents,
                "question": question,
                "web_search": web_search,
                "generation": generation,
                "retry_generation": True,
                "answered_question": False,
                "output": output,
            }
    else:
        output = f"{output}Nope.. I think I'm talking off the top of my head.. just making things up. Let me try again.\n"
        return {
            "documents": documents,
            "question": question,
            "web_search": web_search,
            "generation": generation,
            "retry_generation": True,
            "answered_question": False,
            "output": output,
        }
