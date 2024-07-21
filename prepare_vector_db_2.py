from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma
from constants import APIKEY
import os


os.environ["OPENAI_API_KEY"] = APIKEY

def create_and_save_vector_embedding(): 

    loader = DirectoryLoader("chatbot/data")
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    # index = VectorstoreIndexCreator().from_loaders([loader])    
create_and_save_vector_embedding()