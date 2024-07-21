

from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma
from chatbot.constants import APIKEY
import os
import sys
import openai


model_file = "./chatbot/models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "./chatbot/vectorstores/"

def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type='llama',
        config={'max_new_tokens': 1024,'temperature': 0.01,'context_length': 2048 }
        # max_new_tokens=2048,
        # temperature=0.01,
    )
    return llm

def creat_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt



def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(    
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=1024),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain


def read_vectors_db():
    embedding_model = GPT4AllEmbeddings(model_file="./chatbot/models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model)
    return db


def chatbot(question):
    db = read_vectors_db()
    llm = load_llm(model_file)

    template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
        {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
    prompt = creat_prompt(template)

    llm_chain  =create_qa_chain(prompt, llm, db)
    response = llm_chain.invoke({"query": question})
    
    return response['result']


#------------------------------------------------------------------------------------------

os.environ["OPENAI_API_KEY"] = APIKEY

def chatbot_2(question, chat_history): 
    # if len(sys.argv) > 1:
    #     query = sys.argv[1]
        
    vectorstore = Chroma(persist_directory="./chatbot/persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    
    system_instruction = ""

    chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 5}),
    )
    # chat_history = []
    # print(chat_history)
    result = chain({"question": "Dựa vào thông tin được cung cấp về trường Đại học Khoa học Tự nhiên, Hãy trả lời câu hỏi sau bằng tiếng Việt ( nếu không có thông tin hãy nói rằng không biết, sẽ cập nhật thông tin lại sau). " + question, "chat_history": chat_history})     # chat_history.append((query, result["answer"]))
    return result['answer']

#----------------------------------------NEW VERSION -----------------------------------------------

# def initialize_openai_api(api_key):
#     openai.api_key = api_key

# def convert_chat_history_to_messages(chat_history):
#     messages = []
#     for entry in chat_history:
#         messages.append({"role": "user", "content": entry[0]})
#         messages.append({"role": "assistant", "content": entry[1]})
#     return messages

# def chatbot_2(question, chat_history, api_key):
#     try:
#         initialize_openai_api(api_key)
#         vectorstore = Chroma(persist_directory="./chatbot/persist", embedding_function=openai.Embeddings())
#         index = VectorStoreIndexWrapper(vectorstore=vectorstore)
#         system_instruction = (
#             "Bạn đang nói chuyện với một trợ lý AI chuyên về thông tin "
#             "về trường Đại học Khoa học Tự nhiên. Hãy trả lời các câu hỏi "
#             "bằng tiếng Việt. Nếu không có thông tin, hãy nói rằng không biết "
#             "và sẽ cập nhật thông tin lại sau."
#         )
#         messages = [{"role": "system", "content": system_instruction}]
#         messages.extend(convert_chat_history_to_messages(chat_history))
#         messages.append({"role": "user", "content": question})
#         chain = ConversationalRetrievalChain.from_llm(
#             llm=openai.ChatCompletion(model="gpt-3.5-turbo"),
#             retriever=index.vectorstore.as_retriever(search_kwargs={"k": 5}),
#         )
#         result = chain({
#             "messages": messages
#         })
#         chat_history.append((question, result['choices'][0]['message']['content']))
#         return result['choices'][0]['message']['content']
#     except Exception as e:
#         return f"Đã xảy ra lỗi: {str(e)}"
