from django.db import models

# Create your models here.
class Message(models.Model):
    user_uuid = models.CharField(max_length=500,blank=True,null=True)
    agent = models.CharField(max_length=500,blank=True,null=True)
    question = models.CharField(max_length=500,blank=True,null=True)
    answer = models.TextField(blank=True,null=True)


    def __str__(self):
        return self.question