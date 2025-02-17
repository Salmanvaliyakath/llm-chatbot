from util import SetupIndex
from prompts import rag_prompt_custom, contextualize_q_prompt
from engine import Engine
from langchain.chains import create_history_aware_retriever


selected_model = "qwen2.5:0.5b"
embedding_model = 'nomic-embed-text'

model = Engine(selected_model, embedding_model)
index = SetupIndex(model.embeddings)

retriever = index.create_retriever()

chat_history = list()

def update_chat(chat_history, role, content):
    return chat_history.append({role: content})


while True:

    input_query  = input("Human: ")
    chat_history = update_chat(chat_history, 'Human', input_query)

    history_aware_retriever = create_history_aware_retriever(model.llm_engine, retriever, contextualize_q_prompt)

    retrieved_docs  = retriever.invoke(input_query)
    context         = index.process_text(retrieved_docs)

    prompt    = rag_prompt_custom.invoke({"context": context, "question":input_query})
    response  = model.llm_engine.invoke(prompt)
    chat_history = update_chat(chat_history, 'AI', response)
    
    print("Assistant : ", response.content)
