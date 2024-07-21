import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

import os, json

genai.configure(api_key="AIzaSyDUdM1DzrqW8G04MH9BPfcYRlPsDO7Q0mY")
os.environ["GOOGLE_API_KEY"] = "AIzaSyDUdM1DzrqW8G04MH9BPfcYRlPsDO7Q0mY"

def DRL(info: str) -> str:
    """
    Cung cấp thông tin về điểm rèn luyện của sinh viên.

    Hàm này tìm kiếm thông tin về điểm rèn luyện trong cơ sở dữ liệu dựa trên thông tin đầu vào.

    Args:
        info: Thông tin cần tìm kiếm về điểm rèn luyện. 
             Ví dụ: 
                - "Điểm rèn luyện được cộng khi tham gia câu lạc bộ"
    Returns:
        Chuỗi JSON chứa thông tin về điểm rèn luyện tìm thấy, hoặc chuỗi rỗng nếu không tìm thấy.
    """
    db = FAISS.load_local("googleai_index\\DRL_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    results = db.similarity_search(info, k=10)
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)
    return docs_string

def QCDT(info: str) -> str:
    """
    Cung cấp thông tin về chung về trường đại học Khoa học Tự nhiên và các quy chế liên quan.
    Hàm này tìm kiếm thông tin về trường đại học Khoa học Tự nhiên và các quy chế liên quan trong cơ sở dữ liệu dựa trên thông tin đầu vào.
    Args:
        info: Thông tin cần tìm kiếm về trường đại học Khoa học Tự nhiên. 
             Ví dụ: 
                - "Trường có bao nhiêu cơ sở."
    Returns:
        Chuỗi JSON chứa thông tin về trường đại học Khoa học Tự nhiên tìm thấy, hoặc chuỗi rỗng nếu không tìm thấy.
    """
    db = FAISS.load_local("googleai_index\\QCDT_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    results = db.similarity_search(info, k=10)
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)
    return docs_string

tools = [DRL, QCDT]

available_tools = {
    "DRL": DRL,
    "QCDT": QCDT
}

model = genai.GenerativeModel(model_name="gemini-1.5-flash", tools=tools)
history=[
    {
      "role": "user",
      "parts": [
        """Bạn là một trợ lý ảo thông minh, làm việc cho Trường đại học Khoa học Tự nhiên. 
        Hãy nhớ:
            - Luôn luôn sử dụng các công cụ được cung cấp để tìm kiếm thông tin trước khi đưa ra câu trả lời. 
            - Trả lời một cách ngắn gọn, dễ hiểu. 
            - Không tự ý bịa đặt thông tin. 
            - Sử dụng công cụ phù hợp với nội dung câu hỏi.
            - Nếu không tìm thấy thông tin trong cơ sở dữ liệu, hãy báo cho người dùng.
        """,
      ],
    },
]

chat = model.start_chat(history=history)
while True:
    user_input = input("User: ")

    if user_input.lower() in ["thoát", "exit", "quit"]:
        break
    response = chat.send_message(user_input)
    responses = {}
    for part in response.parts:
        if fn := part.function_call:
            function_name = fn.name
            function_args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
            function_to_call = available_tools[function_name]
            function_response = function_to_call(function_args)
            responses[function_name] = function_response
            print(responses)

    if responses:
        response_parts = [
            genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn, response={"result": val}))
            for fn, val in responses.items()
        ]
        response = chat.send_message(response_parts)
    print("Chatbot:", response.text)