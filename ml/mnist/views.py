from django.shortcuts import render
from django.http import JsonResponse, QueryDict
from matplotlib import pyplot as plt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
import tensorflow as tf

from ml.mnist.fashion_service import FashionService
# Create your views here.
@api_view(['POST'])
@parser_classes([JSONParser])
def fashion(request):
    data = request.data
    test_num = tf.constant(int(data['testNum']))
    result =FashionService().service_model(test_num)
    resp = result
    return JsonResponse({'result':resp})