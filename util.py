from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pathlib import Path


from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate
)

# class Person:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age

#     def greet(self):
#         return f"Hello, my name is {self.name} and I am {self.age} years old."

# from util import Person

# person1 = Person("Salman", 29)
# print(person1.greet())


class SetupIndex:

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.file_path = "Data\Analysis of Actual Fitness Supplement.pdf"
        self.vector_db_path = Path(f"{self.file_path}.faiss")


    def load_pdf_documents(self, file_path):
        document_loader = PyMuPDFLoader(file_path)
        return document_loader.load()

    def chunk_documents(self, raw_documents):
        text_processor = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True
        )
        return text_processor.split_documents(raw_documents)

    def index_documents(self, document_chunks, vector_db_path, embeddings):
        vector_store = FAISS.from_documents(document_chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        vector_store.save_local(str(vector_db_path))

        return retriever

    def process_text(self, retrieved_docs):
        # Combine retrieved content (you could also use a more sophisticated chain)
        return " ".join([doc.page_content for doc in retrieved_docs])

    def create_retriever(self):

        if self.vector_db_path.exists():
            print('Loading data from the disk ...')
            vector_store = FAISS.load_local(str(self.vector_db_path), 
                                            embeddings=self.embeddings,
                                            allow_dangerous_deserialization=True)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        else:

            print('Document loading chunking and indexing ....')
            documents   =   self.load_pdf_documents(self.file_path)
            chunks      =   self.chunk_documents(documents)
            retriever   =   self.index_documents(chunks, self.vector_db_path, self.embeddings)

        print('Retriever loaded')

        return retriever