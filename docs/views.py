from django.shortcuts import render
from rest_framework.decorators import api_view

@api_view(["GET"])
def render_docs(request):
    return render(request, "documentation.html")
