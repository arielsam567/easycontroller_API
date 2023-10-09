from django.http import JsonResponse
from control.matlab import *
import json


def dadosgrafico(request):
    numerador = request.GET.get('numerador')
    denominador = request.GET.get('denominador')
    denominador = denominador.split(',')
    numerador = numerador.split(',')
    numerador2 = []
    denominador2 = []
    for i in numerador:
        numerador2.append(float(i))
    for i in denominador:
        denominador2.append(float(i))
    sys1 = tf(numerador2, denominador2)

    equacao = str(sys1)
    equacao = equacao.split('\n')
    yout, T = step(sys1)
    lista = yout[:]
    lista2 = []
    T2 = []

    for i in lista:
        lista2.append(i)
    for i in T:
        T2.append(i)
    lista2 = json.dumps(lista2)
    lista2 = lista2.split(',')
    T2 = json.dumps(T2)
    T2 = T2.split(',')
    x = T2
    y = lista2

    response = {
        "newY": (y),
        "newX": (x),
        "newEquacao": (equacao),
        "newdenominador": (denominador2),
        "newnumerador": (numerador2)

    }
    return JsonResponse(response, headers={'Access-Control-Allow-Headers': 'Header-Value', 'Access-Control-Allow-Origin': '*'})
