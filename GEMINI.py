import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyDUdM1DzrqW8G04MH9BPfcYRlPsDO7Q0mY"  # Thay thế bằng API key của bạn

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Khởi tạo mô hình nhúng
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Khởi tạo text splitter
text_splitter = RecursiveCharacterTextSplitter(
    # Điều chỉnh các tham số cho phù hợp với tài liệu
    chunk_size = 500,  # Kích thước tối đa cho mỗi đoạn văn bản
    chunk_overlap  = 50,  # Số lượng ký tự chồng chéo giữa các đoạn văn bản
)

# Nhúng drl_new.txt
with open('data\\drl_new.txt', 'r', encoding='utf-8') as f:
    drl_text = f.read()

drl_documents = text_splitter.split_text(drl_text)
drl_db = FAISS.from_texts(drl_documents, embeddings)
drl_db.save_local("googleai_index\\DRL_index")

# Nhúng qcdt_new.txt
with open('data\\qcdt_new.txt', 'r', encoding='utf-8') as f:
    qcdt_text = f.read()

qcdt_documents = text_splitter.split_text(qcdt_text)
qcdt_db = FAISS.from_texts(qcdt_documents, embeddings)
qcdt_db.save_local("googleai_index\\QCDT_index")