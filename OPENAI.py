
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import TextSplitter
from langchain_community.vectorstores import FAISS
import os
os.environ["OPENAI_API_KEY"] = "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"

class LineTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split('\n')

text_splitter = LineTextSplitter()

with open('data\\QUY CHẾ ĐÀO TẠO.txt', 'r', encoding='utf-8') as f:
    text = f.read()

documents = text_splitter.split_text(text)

db = FAISS.from_texts(documents, OpenAIEmbeddings(model="text-embedding-3-large"))
db.save_local("openai_index\\QCDT_index")

with open('data\\ĐIỂM RÈN LUYỆN.txt', 'r', encoding='utf-8') as f:
    text = f.read()

documents = text_splitter.split_text(text)

db = FAISS.from_texts(documents, OpenAIEmbeddings(model="text-embedding-3-large"))
db.save_local("openai_index\\DRL_index")