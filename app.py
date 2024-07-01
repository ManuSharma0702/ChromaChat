from flask import Flask, request, render_template, jsonify, redirect
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: 

{question}


"""


app = Flask(__name__)
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = get_Chat_response(msg)
    return response

def get_Chat_response(query):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query, k=5)
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    # print(prompt)

    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    # Here you can add your logic to generate a response based on the input text
    return response_text
  
if __name__ == '__main__':
    app.run(debug = True)