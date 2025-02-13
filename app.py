
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate
)


# Define a prompt template
prompt_template = """
### System Instruction: You are a Professional Automobile Engineer, Read and understand the following context to answer the upcoming question. Respond in 15 to 20 output tokens.  

---

### Retrieved Context: {context}

---

### User Query: {question}

---

### Response Guidelines:
1. **Prioritize Retrieved Context** - Use the provided context as the primary source of truth. Summarize and synthesize the most relevant information.
2. **Bridge Knowledge Gaps** - If the retrieved context is insufficient, supplement it with your own knowledge, but clearly indicate when you are doing so.
3. **Ensure Accuracy & Clarity** - Provide a clear, structured, and fact-based answer. Avoid speculation.
4. **Cite Sources** - If the retrieved documents contain useful references, include them in your response.
5. **Structured Output** - Format responses in a well-organized manner (e.g., bullet points, step-by-step instructions, or concise summaries).
6. **User-Friendly Language** - Simplify complex concepts when needed, ensuring accessibility for all users.

---

### Output Format:
```json
{{
  "query": "{user_question}",
  "answer": "<Provide a well-structured response here>",
  "sources": ["<List sources from retrieved context, if available>"],
  "additional_notes": "<If retrieved context is insufficient, provide insights based on general knowledge and indicate it>"
}}

---

### Final Response:
(Generate a structured response based on the above instructions.)

"""
# prompt = prompt_template.format(context=context, question=input_query)

rag_prompt_custom = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
    )

selected_model = "qwen2.5:0.5b"
embedding_model = 'nomic-embed-text'

llm_engine  =   ChatOllama(
    model       =   selected_model,
    base_url    =   "http://localhost:11434",
    temperature =   0.3)

embeddings = OllamaEmbeddings(
    model       =embedding_model, 
    base_url    ="http://localhost:11434")

def load_pdf_documents(file_path):
    document_loader = PyMuPDFLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks, vector_db_path):
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    vector_store.save_local(str(vector_db_path))

    return retriever

def process_text(retrieved_docs):
    # Combine retrieved content (you could also use a more sophisticated chain)
    return " ".join([doc.page_content for doc in retrieved_docs])


file_path = "Data\Analysis of Actual Fitness Supplement.pdf"

vector_db_path = Path(f"{file_path}.faiss")

if vector_db_path.exists():
    print('Loading dta from the disk ...')
    vector_store = FAISS.load_local(str(vector_db_path), 
                                    embeddings=embeddings, 
                                    allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

else:

    print('Document loading chunking and indexing ....')
    documents   =   load_pdf_documents(file_path)
    chunks      =   chunk_documents(documents)
    retriever   =   index_documents(chunks, vector_db_path)
    print('Done')


while(1):

    input_query     = input("Human: ")
    retrieved_docs  = retriever.invoke(input_query)
    context = process_text(retrieved_docs)

    prompt = rag_prompt_custom.invoke({"context": context, "question":input_query})

    response    = llm_engine.invoke(prompt)
    print("Assistant : ", response.content)
