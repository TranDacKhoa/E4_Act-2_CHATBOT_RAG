from rest_framework import viewsets, permissions
from .serializers import *
from rest_framework.response import Response
from .models import *
from rest_framework.decorators import action
from .qabot import *
from django.db import connection
from rest_framework import status


class MessageViewset(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    queryset = Message.objects.all()
    serializer_class = MessageSerializer

    def list(self, request):
        queryset = self.queryset
        serializer = self.serializer_class(queryset, many=True)
        return Response(serializer.data)

    def create(self, request):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        else:
            return Response(serializer.errors, status=400)

    def retrieve(self, request, pk=None):
        project = self.queryset.get(pk=pk)
        serializer = self.serializer_class(project)
        return Response(serializer.data)

    def update(self, request, pk=None):
        project = self.queryset.get(pk=pk)
        serializer = self.serializer_class(project, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        else:
            return Response(serializer.errors, status=400)

    def destroy(self, request, pk=None):
        project = self.queryset.get(pk=pk)
        project.delete()
        return Response(status=204)
    
    @action(detail=False, methods=['post'])
    def process_message(self, request):
        question = request.data.get('user_message')
        user_uuid = request.data.get('uuid')

        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM chatbot_message WHERE user_uuid = %s", [user_uuid])
            history_entries = cursor.fetchall()

        chat_history = []  
        # history_entries[0]: id
        # history_entries[1]: question
        # history_entries[2]: answer
        # history_entries[3]: user ip address
        # history_entries[4]: agent
        for entry in history_entries:
            question_his = entry[1]
            answer_his = entry[2]
            chat_history.append((question_his, answer_his))

        # print(entry)
        # print("User IP:", user_uuid)
        # print("User question:", question)
        # print("Chat history:", chat_history)

        # answer = chatbot(question=question, chat_history=chat_history)
        answer = chatbot(question=question)
        # answer = question
        
        response_data = {
            'user_uuid': user_uuid,
            'agent': 'MiniLLama',
            'question': question,
            'answer': answer
        }
        serializer = self.serializer_class(data=response_data)
        if serializer.is_valid():
            serializer.save()

        return Response(response_data)
    

    @action(detail=False, methods=['post'])
    def process_message_gpt(self, request):
        question = request.data.get('user_message')
        user_uuid = request.data.get('uuid')

        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM chatbot_message WHERE user_uuid = %s", [user_uuid])
            history_entries = cursor.fetchall()

        chat_history = []  
        # history_entries[0]: id
        # history_entries[1]: question
        # history_entries[2]: answer
        # history_entries[3]: user_uuid
        # history_entries[4]: agent
        for entry in history_entries:
            question_his = entry[1]
            answer_his = entry[2]
            chat_history.append((question_his, answer_his))

        # print(entry)
        # print("User IP:", user_uuid)
        # print("User question:", question)
        # print("Chat history:", chat_history)

        answer = chatbot_2(question=question, chat_history=chat_history)

        response_data = {
            'user_uuid': user_uuid,
            'agent': 'gpt-4',
            'question': question,
            'answer': answer
        }
        serializer = self.serializer_class(data=response_data)
        if serializer.is_valid():
            serializer.save()

        return Response(response_data)


    @action(detail=False, methods=['delete'])
    def del_chat_history(self, request):
        user_uuid = request.data.get('uuid')
        # print(user_uuid)
        Message.objects.filter(user_uuid=user_uuid).delete()

        return Response("Xoá lịch sử chat thành công", status=status.HTTP_204_NO_CONTENT)

