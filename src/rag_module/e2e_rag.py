import os
import shutil
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from rag_module.sonar import SONAR_Wav_Embeddings
from spoken_chatbot.model import Whisper


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class RAG():
    def __init__(self, args, lan = "en"):
        self.args = args
        self.language = lan
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        self.embedding = self._init_embedding()
        self.persist_directory = self._init_persist_directory()

    def _init_embedding(self):
        if self.args.rag == 'multi':
            return HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large",
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'batch_size': 16, 'normalize_embeddings': False}
            )
        elif self.args.rag == 'bce':
            return HuggingFaceEmbeddings(
                model_name="maidalun1020/bce-embedding-base_v1",
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'batch_size': 16, 'normalize_embeddings': False}
            )
        elif self.args.rag == 'openai':
            api_key = os.environ.get("OPENAI_API_KEY")
            api_base = os.environ.get("OPENAI_API_BASE")
            return OpenAIEmbeddings(openai_api_key=api_key, openai_api_base=api_base)
        else:
            raise ValueError("暂不支持的Embedding类型: {}".format(self.args.rag))
    
    def _init_persist_directory(self):
        return f"answer_data/db_{self.args.rag}_oracle"
    
    def build(self, idx, documents, metadatas = None, search_type = "similarity", search_kwargs = {"k": 4}, reset = True):
        """
        Build the RAG database with the given documents.
        Args:
            idx (int): The index of the database.
            documents (list[str]): The documents to be added to the database.
            metadatas (list[str]): The metadata for the documents. Default is None.
            search_type (str): The type of search to be used. Default is "similarity".
            search_kwargs (dict): Additional arguments for the search. Default is {"k": 4}.
            reset (bool): Whether to reset the database. Default is True.
        """
        docs = []
        if metadatas is not None:
            assert len(documents) == len(metadatas), "documents and metadatas must have the same length"
        else:
            metadatas = [""] * len(documents)
            
        for d, doc in enumerate(documents):
            docs += self.text_splitter.create_documents([doc], metadatas=[{"label": metadatas[d]}])

        db_path = os.path.join(self.persist_directory, f'q{idx}')

        if reset and os.path.exists(db_path):
            shutil.rmtree(db_path)

        if os.path.exists(db_path):
            self.db = Chroma(persist_directory=db_path)
            self.db.add_documents(docs, embedding=self.embedding) 
        else:
            os.makedirs(db_path)
            self.db = Chroma.from_documents(documents=docs, embedding=self.embedding, persist_directory=db_path)

        self.retriever = self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        
    def retrieve(self, query):
        contexts = self.retriever.invoke(query)
        retrieval_context = format_docs(contexts)
        return retrieval_context


class E2E_RAG(RAG):
    def __init__(self, args, lan = "en"):
        super().__init__(args, lan)

    def _init_embedding(self):
        return SONAR_Wav_Embeddings(lan=self.language, device=self.args.device)

    def _init_persist_directory(self):
        return f"answer_data/db_{self.args.rag}"
    
    def retrieve(self, input_path):
        import torchaudio
        query_sr = torchaudio.load(input_path)
        contexts = self.retriever.invoke(query_sr)
        retrieval_context = format_docs(contexts)
        return retrieval_context


class ASR_RAG(RAG):
    def __init__(self, args, lan = "en"):
        super().__init__(args, lan)
        self.asr = Whisper(args)
    
    def _init_persist_directory(self):
        return f"answer_data/db_{self.args.rag}_asr"
    
    def retrieve(self, input_path):
        asr_result = self.asr.audio_to_text(input_path)
        contexts = self.retriever.invoke(asr_result)
        retrieval_context = format_docs(contexts)
        return retrieval_context, asr_result
    

