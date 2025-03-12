from django.urls import path
from .views import render_docs

urlpatterns = [
    path('', render_docs, name="docs_rendered") 
]