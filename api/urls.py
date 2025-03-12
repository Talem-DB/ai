from django.urls import path
from .views import message_ai

urlpatterns = [
    path('', message_ai, name="message_ai") 
]