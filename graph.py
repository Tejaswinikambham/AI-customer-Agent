from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.state import AgentState
from app.retrieval import retrieve_documents
from app.prompts import (
    RELEVANCE_PROMPT,
    REWRITE_PROMPT,
    GENERATION_PROMPT
)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


# NODE 1: Retrieve relevant documents
def retrieve(state: AgentState):

    documents = retrieve_documents(state["question"])

    return {"documents": documents}


# NODE 2: Grade retrieved documents
def grade_documents(state: AgentState):

    documents = state["documents"]

    prompt = ChatPromptTemplate.from_template(
        RELEVANCE_PROMPT
    )

    chain = prompt | llm

    result = chain.invoke({
        "question": state["question"],
        "documents": "\n".join(
            doc.page_content for doc in documents
        )
    })

    relevant = result.content.strip().lower().startswith("yes")

    return {"relevant": relevant}


# NODE 3: Rewrite customer query
def rewrite_query(state: AgentState):

    prompt = ChatPromptTemplate.from_template(
        REWRITE_PROMPT
    )

    chain = prompt | llm

    result = chain.invoke({
        "question": state["question"]
    })

    return {
        "question": result.content.strip(),
        "retries": state["retries"] + 1
    }


# NODE 4: Generate final answer
def generate_answer(state: AgentState):

    context = "\n\n".join(
        doc.page_content for doc in state["documents"]
    )

    prompt = ChatPromptTemplate.from_template(
        GENERATION_PROMPT
    )

    chain = prompt | llm

    result = chain.invoke({
        "question": state["original_question"],
        "context": context if state["relevant"] else ""
    })

    return {"answer": result.content}


# ROUTING LOGIC
def decide_next_step(state: AgentState):

    if state["relevant"]:
        return "generate"

    if state["retries"] < 2:
        return "rewrite"

    return "generate"


# BUILD LANGGRAPH
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("generate", generate_answer)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    decide_next_step,
    {
        "rewrite": "rewrite",
        "generate": "generate"
    }
)

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

agent = workflow.compile()


def run_agent(question: str):

    result = agent.invoke({
        "question": question,
        "original_question": question,
        "documents": [],
        "answer": "",
        "relevant": False,
        "retries": 0
    })

    return result["answer"]