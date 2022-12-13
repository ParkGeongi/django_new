from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from ml.iris.iris_model import IrisModel
from ml.iris.iris_service import IrisService


# Create your views here.
@api_view(['POST'])
@parser_classes([JSONParser])
def iris(request):
    iris_data = request.data
    SepalLengthCm = iris_data['SepalLengthCm']
    SepalWidthCm = iris_data['SepalWidthCm']
    PetalLengthCm = iris_data['PetalLengthCm']
    PetalWidthCm = iris_data['PetalWidthCm']
    print(f'리액트에서 보낸 데이터 : {iris_data}')
    print(f'꽃받침의 길이 : {SepalLengthCm}')
    print(f'꽃받침의 너비 : {SepalWidthCm}')
    print(f'꽃잎의 길이: {PetalLengthCm}')
    print(f'꽃잎의 너비 : {PetalWidthCm}')
    return JsonResponse({'아이리스 결과': 'SUCCESS'})

