
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings


class Engine:

    def __init__(self, selected_model, embedding_model):
        self.llm_engine = self.load_model(selected_model)
        self.embeddings = self.load_embeddings(embedding_model)

    def load_model(self, selected_model):
        model  =   ChatOllama(
            model       =   selected_model,
            base_url    =   "http://localhost:11434",
            temperature =   0.3)
        return model

    def load_embeddings(self, embedding_model):
        embeddings = OllamaEmbeddings(
        model       =embedding_model, 
        base_url    ="http://localhost:11434")

        return embeddings