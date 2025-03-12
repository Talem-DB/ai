from rest_framework.decorators import api_view
from rest_framework.response import Response
from helper.ai import fetch_and_query

@api_view(["POST"])
def message_ai(request):

    form_data = request.data
    query = form_data["query"]

    response = fetch_and_query(query)

    return Response({"response": response})