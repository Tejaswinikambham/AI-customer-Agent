from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma(
    collection_name="customer_support",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


def retrieve_documents(question: str):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    documents = retriever.invoke(question)

    return documents