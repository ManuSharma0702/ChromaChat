from langchain.document_loaders import WebBaseLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import  OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
from get_embedding_function import get_embedding_function
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

url="https://brainlox.com/courses/category/technical"
#LOADING DATA
def load_data(url):
    loader=WebBaseLoader([url])
    tt=Html2TextTransformer()
    docs=tt.transform_documents(loader.load())
    ts=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=0)
    fd=ts.split_documents(docs)
    print(len(fd))
    return fd
    
def add_to_chroma(chunks):
    embedding_function = get_embedding_function()
    print(f"👉 Adding new documents:")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    db.add_documents(chunks)
    db.persist()

def main():
    url="https://brainlox.com/courses/category/technical"
    chunks = load_data(url)
    add_to_chroma(chunks)
    
if __name__ == "__main__":
    main()
