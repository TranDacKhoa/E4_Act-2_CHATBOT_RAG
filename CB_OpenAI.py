from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import json, os 

os.environ['OPENAI_API_KEY'] = "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"
client = OpenAI()

def DRL(info):
    db = FAISS.load_local("openai_index/DRL_index", OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization=True)
    results = db.similarity_search(info, k=5)
    if not results:
        return None
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)
    return docs_string

def QCDT(info):
    db = FAISS.load_local("openai_index/QCDT_index", OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization=True)
    results = db.similarity_search(info, k=5)
    if not results:
        return None
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)
    return docs_string

available_functions = {
    "DRL": DRL,
    "QCDT": QCDT
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "DRL",
            "description": "Cung cấp những thông tin liên quan đến điểm rèn luyện mà bạn cần biết.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {
                        "type": "string",
                        "description": "Thông tin mà bạn cần tìm kiếm, e.g. Chấp hành và không vi phạm: +15đ rèn luyện ",
                    },
                },
                "required": ["info"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "QCDT",
            "description": "Cung cấp những thông tin liên quan đến trường mà bạn cần biết.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {
                        "type": "string",
                        "description": "Thông tin mà bạn cần tìm kiếm, e.g. Trường có 2 cơ sở",
                    },
                },
                "required": ["info"],
            },
        },
    },
]

memory = [
    {
        "role": "system", 
        "content": """Bạn là trợ lý ảo thông minh làm việc cho trường đại học Khoa học Tự Nhiên, được cung cấp những công cụ truy xuất dữ liệu để trả lời câu hỏi của người dùng. \n
                     IMPORTANT: LUÔN LUÔN PHẢI tìm thông tin trong các tài liệu bằng tools được cung cấp trước khi trả lời câu hỏi của người dùng!"""
    },
]

def chat_completion_request(messages, functions=None, model="gpt-4"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=functions,
            tool_choice="auto", 
            temperature=0,
        )
        print(response)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        

        if tool_calls:
            messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                if function_name in available_functions:
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(function_args.get("info"))
                    if function_response:
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": "Xin lỗi, tôi không thể tìm thấy thông tin bạn yêu cầu.",
                            }
                        )
            return chat_completion_request(messages=messages, functions=functions)
        else:
            msg = response_message.content
            return msg
        
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
    
if __name__ == "__main__":
    print("Bắt đầu trò chuyện với trợ lý ảo (nhập 'exit' để dừng)")
    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break
        memory.append({"role": "user", "content": query})
        response = chat_completion_request(messages=memory, functions=tools)
        print(f"Chatbot: {response}")
        memory.append({"role": "assistant", "content": response})
