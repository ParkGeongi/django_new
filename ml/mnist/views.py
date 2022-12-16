import json

from django.shortcuts import render
from django.http import JsonResponse, QueryDict
from django.utils.datastructures import MultiValueDictKeyError
from matplotlib import pyplot as plt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
import tensorflow as tf
from rest_framework.response import Response
from ml.mnist.fashion_service import FashionService
# Create your views here.
@api_view(["GET","POST"])
def fashion(request):
    if request.method == 'GET':
        # body = request.body  # byte string of JSON data
        # print(f" ######## request.body is {body} ########  ")
        # data = json.loads(body)  # json to dict
        # print(request.headers)  # request의 header 정보
        # print(request.content_type)  # application/json
        print(f"######## GET at Here React ID is {int(request.GET['Num'])} ########")
        #return JsonResponse(
            #{'result': FashionService().service_model(int(request.GET['Num']))})


        return JsonResponse({'result': FashionService().service_model(int(request.GET['Num']))})

    elif request.method == 'POST':

        #req = request.body['num']
        #req = data['Num']
        data = int(request.GET['Num'])
        print(f'############################{data}')

        return JsonResponse({'result': 'success'})
        #data = json.loads(request)  # json to dict
        #print(f"######## POST at Here ! React ID is {data['POSTNum']} ########")

        #return JsonResponse({'result': FashionService().service_model(int(data['POSTNum']))})


