
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.views import View
from django.views.generic import TemplateView
from django.shortcuts import render

import control
import control.matlab as conmat
import logging
import json
import numpy as np
from decimal import *
import collections.abc
import time


class Inicio(TemplateView):
    template_name = "templates/inicio.html"


def HomeView(request):
    return render(request, 'templates/home.html', {})


def IndexView(request):
    return render(request, 'templates/index.html', {})


def CadastroView(request):
    return render(request, 'templates/cadastro.html', {})


def SobreView(request):
    return render(request, 'templates/sobre.html', {})


def TutorialView(request):
    return render(request, 'templates/tutorial.html', {})


def serve_angular(request):
    return render(request, 'time-controller-domain.component.html')


"""---------------------------------------------Controlador LQR---------------------------------------------"""


@method_decorator(csrf_exempt, name='dispatch')
class EasyControllerLqr(View):
    logging.basicConfig(level=logging.DEBUG)

    def options(self, request, *args, **kwargs):
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': '*'
        }
        return JsonResponse({}, status=200, safe=True, headers=headers)

    def post(self, request):
        class NumpyArrayEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        data = json.loads(request.body)
        matrizQ = np.array(data.get('Q'), dtype=float)
        matrizR = np.array(data.get('R'), dtype=float)
        matrizA = np.array(data.get('A'), dtype=float)
        matrizB = np.array(data.get('B'), dtype=float)
        matrizC = np.array(data.get('C'), dtype=float)
        matrizD = np.array(data.get('D'), dtype=float)
        ci = np.array(data.get('CI'), dtype=float)
        body = data.get('getbody', {})

        saturacao = np.array(data.get('SAT'), dtype=float)
        saturacao1 = body.get('saturacao1')
        saturacao1 = float(saturacao1)

        if (saturacao1 == 1):
            lim_sup = saturacao[0, :]
            lim_inf = saturacao[1, :]

        T = body.get('amostragem')
        T = float(T)

        [K, S, E] = control.lqr(matrizA, matrizB, matrizQ, matrizR)
        auto_val = np.linalg.eigvals(matrizA - matrizB * K)
        min_auto = min(abs(auto_val))
        max_auto = max(abs(auto_val))
        constMax = (max_auto/min_auto)

        if (T == 0):
            T = (2 * np.pi/(10000 * min_auto))

        if max_auto == min_auto:
            constMax = constMax * 2
        Tmax = ((2 * np.pi)/(max_auto)) * (constMax)

        t = np.arange(0, Tmax + T, T)

        Ni = len(t)
        Nx = len(matrizA)
        Nu = np.size(matrizB, 1)
        Ny = np.size(matrizC, 0)

        u = np.zeros((Nu, Nx))
        if Nx == 2:
            x = (ci).T * np.ones((Nx, Nx))
        else:
            x = (ci).T * np.ones((Nx, Nx))
        y = np.zeros((Ny, Nu))

        for k in range(Nx, Ni):

            dx = matrizA @ np.transpose([x[:, k - 1]]) + \
                matrizB @ np.transpose([u[:, k - 1]])
            dx = np.reshape(dx, (Nx, 1))
            x_linha = np.transpose([x[:, k - 1]]) + dx * T
            x_linha = np.reshape(x_linha, (Nx, 1))
            x = np.concatenate((x, x_linha), axis=1)
            y = np.concatenate((y, matrizC @ np.transpose([x[:, k]])), axis=1)
            u_linhaA = np.transpose([x[:, k - 1]])
            u_linhaA = np.reshape(u_linhaA, (Nx, 1))
            u_linha = - K * u_linhaA
            u = np.concatenate((u, u_linha), axis=1)

            if (saturacao1 == 1):
                for i in range(u.shape[0]):
                    linha = u[i]
                    valorsup = lim_sup[i]
                    valorinf = lim_inf[i]
                    u[i] = np.where(linha > valorsup, valorsup, np.where(linha < valorinf, valorinf, linha))

        yRavel = np.ravel(y)
        ySplit = np.split(yRavel, Ny)

        encodeY = json.dumps(ySplit, cls=NumpyArrayEncoder)
        encodeU = json.dumps(u, cls=NumpyArrayEncoder)
        encodeK = json.dumps(K, cls=NumpyArrayEncoder)
        encodeNx = json.dumps(Nx, cls=NumpyArrayEncoder)
        encodeNu = json.dumps(Nu, cls=NumpyArrayEncoder)
        encodeNy = json.dumps(Ny, cls=NumpyArrayEncoder)
        encodeT = json.dumps(T, cls=NumpyArrayEncoder)

        reponse = {
            "Yout": encodeY,
            "time": t.tolist(),
            "Uhat": encodeU,
            "K": encodeK,
            "Nx": encodeNx,
            "Nu": encodeNu,
            "Ny": encodeNy,
            "T": encodeT,
        }

        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': '*'
        }
        response = JsonResponse(reponse, status=201,
                                safe=True, headers=headers)
        return response


"""---------------------------------------------Controlador LQI---------------------------------------------"""


@method_decorator(csrf_exempt, name='dispatch')
class EasyControllerLqi(View):
    logging.basicConfig(level=logging.DEBUG)

    def options(self, request, *args, **kwargs):
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': '*'
        }
        return JsonResponse({}, status=200, safe=True, headers=headers)

    def post(self, request):
        try:
            class NumpyArrayEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    return json.JSONEncoder.default(self, obj)

            data = json.loads(request.body)
            matrizQ = np.array(data.get('Q'), dtype=float)
            matrizR = np.array(data.get('R'), dtype=float)
            matrizA = np.array(data.get('A'), dtype=float)
            matrizB = np.array(data.get('B'), dtype=float)
            matrizC = np.array(data.get('C'), dtype=float)
            matrizD = np.array(data.get('D'), dtype=float)
            ci = 0
            body = data.get('getbody', {})

            saturacao = np.array(data.get('SAT'), dtype=float)
            saturacao1 = body.get('saturacao1')
            saturacao1 = float(saturacao1)

            if (saturacao1 == 1):
                lim_sup = saturacao[0, :]
                lim_inf = saturacao[1, :]

            T = body.get('amostragem')
            T = float(T)

            referencia = np.array(data.get('REF'), dtype=float)
            referencia1 = body.get('referencia1')
            referencia1 = float(referencia1)

            Ahat = np.concatenate(((np.concatenate((matrizA, np.zeros((np.size(matrizA, 0), np.size(matrizC, 0)))), axis=1)), (
                np.concatenate((-matrizC, np.zeros((np.size(matrizC, 0), np.size(matrizC, 0)))), axis=1))), axis=0)

            Bhat = np.concatenate(
                (matrizB, np.zeros((np.size(matrizC, 0), np.size(matrizB, 1)))), axis=0)

            Chat = np.concatenate(
                (matrizC, np.zeros((np.size(matrizC, 0), 1))), axis=1)

            Dhat = np.zeros((np.size(Chat, 0), np.size(Bhat, 1)))

            Nx = len(matrizA)
            Nu = np.size(matrizB, 1)
            Ny = np.size(matrizC, 0)

            [Ke, S, E] = control.lqr(Ahat, Bhat, matrizQ, matrizR)
            Ki = Ke[0:Nu, Nx:Nx+Ny]
            K = Ke[:, 0:Nx]

            ssResponse = control.StateSpace(
                        Ahat-Bhat*Ke, Bhat, Chat, Dhat)

            [wn, zeta, poles] = control.damp(ssResponse)
            zeta[zeta != zeta] = 1
            for i in wn:
                timeconst = 1/(zeta*wn)
                # timeconst = timeconst.round()
            if max(timeconst) == float("inf"):
                newtimeconst = np.delete(
                    timeconst, np.where(timeconst == float("inf")))
                if len(newtimeconst) == 0:
                    newtimeconst = np.arange(3)
                timeSim = 5*max(newtimeconst)
            else:
                timeSim = 5*max(timeconst)

            if (T == 0):
                T = timeSim/10000
            t = np.arange(0, timeSim, T, dtype=float)

            Ni = len(t)
            Nx = len(matrizA)
            Nu = np.size(matrizB, 1)
            Ny = np.size(matrizC, 0)

            u = np.zeros((Nu, Nx))
            x = (ci) * np.ones((Nx, Nx))

            if (referencia1 == 1):
                ref = np.tile(referencia, Ni)
            else:
                ref = np.ones((Ny, Ni))

            y = np.zeros((Ny, Nx))
            int_e = 0

            # if Nu == Ny:
            #     Ki = Ki.T

            tempo_limite = 10.0
            inicio_tempo = time.perf_counter()

            for k in range(Nx, Ni):

                tempo_decorrido = time.perf_counter() - inicio_tempo
                if tempo_decorrido > tempo_limite:
                    raise TimeoutError(f"O cálculo está demorando mais de {tempo_limite} segundos.")
                
                dx = matrizA @ np.transpose([x[:, k - 1]]) + \
                    matrizB @ np.transpose([u[:, k - 1]])
                dx = np.reshape(dx, (Nx, 1))
                x_linha = np.transpose([x[:, k - 1]]) + dx * T
                x_linha = np.reshape(x_linha, (Nx, 1))
                x = np.concatenate((x, x_linha), axis=1)
                y = np.concatenate((y, matrizC @ np.transpose([x[:, k]])), axis=1)

                e = np.transpose([ref[:, k - 1] - y[:, k - 1]])
                int_e = int_e + e * T
                u_linhaA = np.transpose([x[:, k - 1]])
                u_linha = - K * u_linhaA - Ki * int_e
                u = np.concatenate((u, u_linha), axis=1)

                if (saturacao1 == 1):
                    for i in range(u.shape[0]):
                        linha = u[i]
                        valorsup = lim_sup[i]
                        valorinf = lim_inf[i]
                        u[i] = np.where(linha > valorsup, valorsup, np.where(linha < valorinf, valorinf, linha))

            yRavel = np.ravel(y)
            ySplit = np.split(yRavel, Ny)

            encodeY = json.dumps(ySplit, cls=NumpyArrayEncoder)
            encodeU = json.dumps(u, cls=NumpyArrayEncoder)
            encodeK = json.dumps(K, cls=NumpyArrayEncoder)
            encodeKi = json.dumps(Ki, cls=NumpyArrayEncoder)
            encodeNx = json.dumps(Nx, cls=NumpyArrayEncoder)
            encodeNu = json.dumps(Nu, cls=NumpyArrayEncoder)
            encodeNy = json.dumps(Ny, cls=NumpyArrayEncoder)
            encodeT = json.dumps(T, cls=NumpyArrayEncoder)

            reponse = {
                "Yout": encodeY,
                "time": t.tolist(),
                "Uhat": encodeU,
                "K": encodeK,
                "Ki": encodeKi,
                "Nx": encodeNx,
                "Nu": encodeNu,
                "Ny": encodeNy,
                "T": encodeT,
            }

            headers = {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': '*'
            }
            response = JsonResponse(reponse, status=201,
                                    safe=True, headers=headers)
            return response
        
        except TimeoutError as e:
            return JsonResponse({"success": False, "error": str(e)}, status=500)

"""---------------------------------------------Controlador LQG---------------------------------------------"""


@method_decorator(csrf_exempt, name='dispatch')
class EasyControllerLqg(View):
    logging.basicConfig(level=logging.DEBUG)

    def options(self, request, *args, **kwargs):
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': '*'
        }
        return JsonResponse({}, status=200, safe=True, headers=headers)

    def post(self, request):
        class NumpyArrayEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                return json.JSONEncoder.default(self, obj)

        data = json.loads(request.body)
        matrizQ = np.array(data.get('Q'), dtype=float)
        matrizR = np.array(data.get('R'), dtype=float)
        matrizQN = np.array(data.get('QN'), dtype=float)
        matrizRN = np.array(data.get('RN'), dtype=float)
        matrizA = np.array(data.get('A'), dtype=float)
        matrizB = np.array(data.get('B'), dtype=float)
        matrizC = np.array(data.get('C'), dtype=float)
        matrizD = np.array(data.get('D'), dtype=float)
        ci = np.array(data.get('CI'), dtype=float)
        body = data.get('getbody', {})

        saturacao = np.array(data.get('SAT'), dtype=float)
        saturacao1 = body.get('saturacao1')
        saturacao1 = float(saturacao1)

        if (saturacao1 == 1):
            lim_sup = saturacao[0, :]
            lim_inf = saturacao[1, :]

        T = body.get('amostragem')
        T = float(T)

        if np.size(matrizB, 1) or np.size(matrizC, 0) > 1:
            Dhat = np.zeros((np.size(matrizC, 0), np.size(matrizB, 1)))
        else:
            Dhat = matrizD

        [K, S, E] = control.lqr(matrizA, matrizB, matrizQ, matrizR)

        auto_val = np.linalg.eigvals(matrizA - matrizB * K)
        min_auto = min(abs(auto_val))
        max_auto = max(abs(auto_val))
        if (T == 0):
            T = (2 * np.pi/(10000 * min_auto))
        constMax = (max_auto/min_auto)
        if max_auto == min_auto:
            constMax = constMax * 4
        Tmax = ((2 * np.pi)/(max_auto)) * (constMax)
        t = np.arange(0, Tmax + T, T)

        Ni = len(t)
        Nx = len(matrizA)
        Nu = np.size(matrizB, 1)
        Ny = np.size(matrizC, 0)

        u = np.zeros((Nu, Nx))
        if Nx == 2:
            x = np.transpose(ci) * np.ones((Nx, Nx))
        else:
            x = (ci).T * np.ones((Nx, Nx))
        xhat = np.zeros((Nx, Nx))
        y = np.zeros((Ny, Nx))

        sys = control.StateSpace(matrizA, matrizB, matrizC, Dhat)
        [L, P, EK] = control.lqe(sys, matrizQN, matrizRN)

        for k in range(Nx, Ni):

            dx = matrizA @ np.transpose([x[:, k - 1]]) + \
                matrizB @ np.transpose([u[:, k - 1]])
            dx = np.reshape(dx, (Nx, 1))
            x_linha = np.transpose([x[:, k - 1]]) + dx * T
            x_linha = np.reshape(x_linha, (Nx, 1))
            x = np.concatenate((x, x_linha), axis=1)
            y = np.concatenate((y, matrizC @ np.transpose([x[:, k]])), axis=1)

            dx_hatA = np.transpose([xhat[:, k - 1]])
            dx_hatA = np.reshape(dx_hatA, (Nx, 1))
            dx_hatB = matrizB @ np.transpose([u[:, k - 1]])
            dx_hatB = np.reshape(dx_hatB, (Nx, 1))
            dx_hat = (matrizA - L * matrizC) @ dx_hatA + \
                dx_hatB + L * np.transpose([y[:, k-1]])

            xhat_linhaA = np.transpose([xhat[:, k - 1]])
            xhat_linhaA = np.reshape(xhat_linhaA, (Nx, 1))
            xhat_linha = xhat_linhaA + dx_hat * T
            xhat = np.concatenate((xhat, xhat_linha), axis=1)

            u_linhaA = np.transpose([xhat[:, k - 1]])
            u_linhaA = np.reshape(u_linhaA, (Nx, 1))
            u_linha = - K * u_linhaA
            u = np.concatenate((u, u_linha), axis=1)

            if (saturacao1 == 1):
                for i in range(u.shape[0]):
                    linha = u[i]
                    valorsup = lim_sup[i]
                    valorinf = lim_inf[i]
                    u[i] = np.where(linha > valorsup, valorsup, np.where(linha < valorinf, valorinf, linha))

        encodeY = json.dumps(y, cls=NumpyArrayEncoder)
        encodeU = json.dumps(u, cls=NumpyArrayEncoder)
        encodeK = json.dumps(K, cls=NumpyArrayEncoder)
        encodeL = json.dumps(L, cls=NumpyArrayEncoder)
        encodeNx = json.dumps(Nx, cls=NumpyArrayEncoder)
        encodeNu = json.dumps(Nu, cls=NumpyArrayEncoder)
        encodeNy = json.dumps(Ny, cls=NumpyArrayEncoder)
        encodeT = json.dumps(T, cls=NumpyArrayEncoder)

        reponse = {
            "Yout": encodeY,
            "time": t.tolist(),
            "Uhat": encodeU,
            "K": encodeK,
            "L": encodeL,
            "Nx": encodeNx,
            "Nu": encodeNu,
            "Ny": encodeNy,
            "T": encodeT,
        }

        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': '*'
        }
        response = JsonResponse(reponse, status=201,
                                safe=True, headers=headers)
        return response


"""---------------------------------------------Controlador LQGI---------------------------------------------"""


@method_decorator(csrf_exempt, name='dispatch')
class EasyControllerLqgi(View):
    logging.basicConfig(level=logging.DEBUG)

    def options(self, request, *args, **kwargs):
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': '*'
        }
        return JsonResponse({}, status=200, safe=True, headers=headers)

    def post(self, request):
        class NumpyArrayEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                return json.JSONEncoder.default(self, obj)

        data = json.loads(request.body)
        matrizQ = np.array(data.get('Q'), dtype=float)
        matrizR = np.array(data.get('R'), dtype=float)
        matrizQN = np.array(data.get('QN'), dtype=float)
        matrizRN = np.array(data.get('RN'), dtype=float)
        matrizA = np.array(data.get('A'), dtype=float)
        matrizB = np.array(data.get('B'), dtype=float)
        matrizC = np.array(data.get('C'), dtype=float)
        matrizD = np.array(data.get('D'), dtype=float)
        ci = 0
        body = data.get('getbody', {})

        saturacao = np.array(data.get('SAT'), dtype=float)
        saturacao1 = body.get('saturacao1')
        saturacao1 = float(saturacao1)

        if (saturacao1 == 1):
            lim_sup = saturacao[0, :]
            lim_inf = saturacao[1, :]

        T = body.get('amostragem')
        T = float(T)

        referencia = np.array(data.get('REF'), dtype=float)
        referencia1 = body.get('referencia1')
        referencia1 = float(referencia1)

        Ahat = np.concatenate(((np.concatenate((matrizA, np.zeros((np.size(matrizA, 0), np.size(matrizC, 0)))), axis=1)), (
            np.concatenate((-matrizC, np.zeros((np.size(matrizC, 0), np.size(matrizC, 0)))), axis=1))), axis=0)
        Bhat = np.concatenate(
            (matrizB, np.zeros((np.size(matrizC, 0), np.size(matrizB, 1)))), axis=0)
        Chat = np.concatenate(
            (matrizC, np.zeros((np.size(matrizC, 0), 1))), axis=1)

        Dhat = np.zeros((np.size(matrizC, 0), np.size(matrizB, 1)))

        Nx = len(matrizA)
        Nu = np.size(matrizB, 1)
        Ny = np.size(matrizC, 0)

        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': '*'
        }

        [Ke, S, E] = control.lqr(Ahat, Bhat, matrizQ, matrizR)
        Ki = Ke[0:Nu, Nx:Nx+Ny]
        K = Ke[:, 0:Nx]

        ssResponse = control.StateSpace(
                    Ahat-Bhat*Ke, Bhat, Chat, Dhat)

        [wn, zeta, poles] = control.damp(ssResponse)
        zeta[zeta != zeta] = 1
        for i in wn:
            timeconst = 1/(zeta*wn)
            # timeconst = timeconst.round()
        if max(timeconst) == float("inf"):
            newtimeconst = np.delete(
                timeconst, np.where(timeconst == float("inf")))
            if len(newtimeconst) == 0:
                newtimeconst = np.arange(3)
            timeSim = 5*max(newtimeconst)
        else:
            timeSim = 5*max(timeconst)

        if (T == 0):
            T = timeSim/10000
        t = np.arange(0, timeSim, T, dtype=float)

        Ni = len(t)

        u = np.zeros((Nu, Nx))
        x = np.transpose(ci) * np.ones((Nx, Nx))
        xhat = np.zeros((Nx, Nx))
        if (referencia1 == 1):
            ref = np.tile(referencia, Ni)
        else:
            ref = np.ones((Ny, Ni))

        int_e = 0

        y = np.zeros((Ny, Nx))

        # if Nu == Ny:
        #     Ki = Ki.T

        sys = control.StateSpace(matrizA, matrizB, matrizC, Dhat)
        [L, P, EK] = control.lqe(sys, matrizQN, matrizRN)

        for k in range(Nx, Ni):

            dx = matrizA @ np.transpose([x[:, k - 1]]) + \
                matrizB @ np.transpose([u[:, k - 1]])
            dx = np.reshape(dx, (Nx, 1))
            x_linha = np.transpose([x[:, k - 1]]) + dx * T
            x_linha = np.reshape(x_linha, (Nx, 1))
            x = np.concatenate((x, x_linha), axis=1)

            y = np.concatenate((y, matrizC @ np.transpose([x[:, k]])), axis=1)

            dx_hatA = np.transpose([xhat[:, k - 1]])
            dx_hatA = np.reshape(dx_hatA, (Nx, 1))
            dx_hatB = matrizB @ np.transpose([u[:, k - 1]])
            dx_hatB = np.reshape(dx_hatB, (Nx, 1))
            dx_hat = (matrizA - L * matrizC) @ dx_hatA + \
                dx_hatB + L * np.transpose([y[:, k-1]])

            xhat_linhaA = np.transpose([xhat[:, k - 1]])
            xhat_linhaA = np.reshape(xhat_linhaA, (Nx, 1))
            xhat_linha = xhat_linhaA + dx_hat * T
            xhat = np.concatenate((xhat, xhat_linha), axis=1)

            e = np.transpose([ref[:, k - 1] - y[:, k - 1]])
            int_e = int_e + e * T
            u_linhaA = np.transpose([xhat[:, k - 1]])
            u_linhaA = np.reshape(u_linhaA, (Nx, 1))
            u_linha = - K * u_linhaA - Ki * int_e
            u = np.concatenate((u, u_linha), axis=1)

            if (saturacao1 == 1):
                for i in range(u.shape[0]):
                    linha = u[i]
                    valorsup = lim_sup[i]
                    valorinf = lim_inf[i]
                    u[i] = np.where(linha > valorsup, valorsup, np.where(linha < valorinf, valorinf, linha))

        encodeY = json.dumps(y, cls=NumpyArrayEncoder)
        encodeU = json.dumps(u, cls=NumpyArrayEncoder)
        encodeK = json.dumps(K, cls=NumpyArrayEncoder)
        encodeKi = json.dumps(Ki, cls=NumpyArrayEncoder)
        encodeL = json.dumps(L, cls=NumpyArrayEncoder)
        encodeNx = json.dumps(Nx, cls=NumpyArrayEncoder)
        encodeNu = json.dumps(Nu, cls=NumpyArrayEncoder)
        encodeNy = json.dumps(Ny, cls=NumpyArrayEncoder)
        encodeT = json.dumps(T, cls=NumpyArrayEncoder)

        reponse = {
            "Yout": encodeY,
            "time": t.tolist(),
            "Uhat": encodeU,
            "K": encodeK,
            "Ki": encodeKi,
            "L": encodeL,
            "Nx": encodeNx,
            "Nu": encodeNu,
            "Ny": encodeNy,
            "T": encodeT,

        }

        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': '*'
        }
        response = JsonResponse(reponse, status=201,
                                safe=True, headers=headers)
        return response


@method_decorator(csrf_exempt, name='dispatch')
class EasyController(View):
    logging.basicConfig(level=logging.DEBUG)

    def options(self, request, *args, **kwargs):
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': '*'
        }
        return JsonResponse({}, status=200, safe=True, headers=headers)

    def post(self, request):
        data = json.loads(request.body)
        matrizA = np.array(data.get('A'))
        matrizB = np.array(data.get('B'))
        matrizC = np.array(data.get('C'))
        matrizD = np.zeros((1, 1))

        # logging.info('Dados das matrizA >> %s', matrizA)
        # logging.info('Dados das matrizB >> %s', matrizB)
        # logging.info('Dados das matrizC >> %s', matrizC)
        # logging.info('Dados das matrizD >> %s', matrizD)

        cols = len(matrizB[0])
        # logging.info('cols >> %s', cols)
        matrizB_cols = np.split(matrizB, cols, axis=1)
        # logging.info('matrizB_cols >> %s', matrizB_cols)
        outY_total = []

        # logging.info('---------------Dinâmica da Planta---------------')

        for n in range(cols):

            # logging.info('i >> %s', matrizB_cols[n])

            for c in matrizC:
                # logging.info('n >> %s', c)

                ssResponse = control.StateSpace(
                    matrizA, matrizB_cols[n], c, matrizD)

                [wn, zeta, poles] = control.damp(ssResponse)
                zeta[zeta != zeta] = 1
                for i in wn:
                    timeconst = 1/(zeta*wn)
                    # timeconst = timeconst.round()
                if max(timeconst) == float("inf"):
                    newtimeconst = np.delete(
                        timeconst, np.where(timeconst == float("inf")))
                    if len(newtimeconst) == 0:
                        newtimeconst = np.arange(3)
                    timeSim = 5 * max(newtimeconst)
                else:
                    timeSim = 5 * max(timeconst)
                time = np.arange(0, timeSim, timeSim/10000, dtype=float)

                out_y, time, out_x = conmat.lsim(ssResponse, 1, time)

                stepY = []
                for i in out_y:
                    if isinstance(i, collections.abc.Sequence):
                        stepY.append(float("{:.4f}".format(i[0])))
                    else:
                        stepY.append(float("{:.4f}".format(i)))

                outY_total.append(stepY)

        ctrb = conmat.ctrb(matrizA, matrizB)
        control_rank = np.linalg.matrix_rank(ctrb)
        control_rank_result = "O sistema não é controlável"
        if control_rank >= len(matrizA):
            control_rank_result = "O sistema é controlável"

        obsv = conmat.obsv(matrizA, matrizC)
        obsv_rank = np.linalg.matrix_rank(obsv)
        obsv_rank_result = "O sistema não é observável"
        if obsv_rank >= len(matrizA):
            obsv_rank_result = "O sistema é observável"

        reponse = {
            "StateSpace": {
                "A": ssResponse.A.tolist(),
                "B": ssResponse.B.tolist(),
                "C": ssResponse.C.tolist(),
                "D": ssResponse.D.tolist(),
            },
            "stepY": stepY,
            "outY_total": outY_total,
            "time": time.tolist(),
            "control_rank": control_rank_result,
            "obsv_rank": obsv_rank_result
        }
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': '*'
        }
        response = JsonResponse(reponse, status=201,
                                safe=True, headers=headers)
        return response
