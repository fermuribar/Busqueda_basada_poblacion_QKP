import numpy as np
import config as conf
import matplotlib.pyplot as plt

#  _               _                      __ _      _                    
# | |             | |                    / _(_)    | |                   
# | |     ___  ___| |_ _   _ _ __ __ _  | |_ _  ___| |__   ___ _ __ ___  
# | |    / _ \/ __| __| | | | '__/ _` | |  _| |/ __| '_ \ / _ \ '__/ _ \ 
# | |___|  __/ (__| |_| |_| | | | (_| | | | | | (__| | | |  __/ | | (_) |
# |______\___|\___|\__|\__,_|_|  \__,_| |_| |_|\___|_| |_|\___|_|  \___/ 

# Función para leer el archivo y procesar los datos
def procesar_archivo(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        lineas = archivo.readlines()
        
        # Tamaño de la matriz (segunda línea del archivo)
        tamaño = int(lineas[1].strip())
        matriz = np.zeros((tamaño, tamaño))
        
        # Rellenar la diagonal principal
        diagonal_principal = [int(x) for x in lineas[2].split()]
        np.fill_diagonal(matriz, diagonal_principal)
        
        # Rellenar el triángulo superior
        fila_actual = 3
        for i in range(tamaño):
            if fila_actual >= len(lineas) or lineas[fila_actual].strip() == '':
                    break
            valores = [int(x) for x in lineas[fila_actual].split()]
            for j in range(i+1, tamaño):
                if valores:
                    matriz[i, j] = valores.pop(0)
                    matriz[j, i] = matriz[i, j]
            
            fila_actual += 1

        # Buscar el peso y el último vector después de la línea en blanco
        fila_actual += 2
        peso = int(lineas[fila_actual].strip())
        vector_pesos = np.array([int(x) for x in lineas[fila_actual+1].split()])

    return matriz, peso, vector_pesos

#  __  __ ______ _______       _    _ ______ _    _ _____  _____  _____ _______ _____ _____          
# |  \/  |  ____|__   __|/\   | |  | |  ____| |  | |  __ \|_   _|/ ____|__   __|_   _/ ____|   /\    
# | \  / | |__     | |  /  \  | |__| | |__  | |  | | |__) | | | | (___    | |    | || |       /  \   
# | |\/| |  __|    | | / /\ \ |  __  |  __| | |  | |  _  /  | |  \___ \   | |    | || |      / /\ \  
# | |  | | |____   | |/ ____ \| |  | | |____| |__| | | \ \ _| |_ ____) |  | |   _| || |____ / ____ \ 
# |_|  |_|______|  |_/_/    \_\_|  |_|______|\____/|_|  \_\_____|_____/   |_|  |_____\_____/_/    \_\

#Clase Solucion
class Solucion:
    def __init__(self) -> None:
        self.solucion = None
        self.peso = None
        self.beneficio = None


class Vecindarios:
    def __init__(self,solucion, vecindario = 0) ->np.ndarray:
        self.solucion_generadora = solucion.copy()
        
        # 1. Índices de los "1"
        indices_1 = np.where(solucion == 1)[0]

        # 2. Índices de los "0"
        indices_0 = np.where(solucion == 0)[0]
        
        a_repetido = np.repeat(indices_1, len(indices_0), axis=0)
        b_tile = np.tile(indices_0, len(indices_1))
        duplas = np.column_stack((a_repetido, b_tile))

        if vecindario != 0:
            a_repetido_d = np.repeat(duplas, len(indices_0), axis=0)
            b_tile_d = np.tile(indices_0, len(duplas))

            permutaciones = np.column_stack((a_repetido_d, b_tile_d))
        
            self.permutaciones = (permutaciones[permutaciones[:, 1] != permutaciones[:, 2]]).copy()
        else:
            self.permutaciones = duplas.copy()

        self.indice_permutacion = np.random.permutation(np.arange(0, self.permutaciones.shape[0]))

        self.vector_elementos_sin_explorar = np.ones(self.permutaciones.shape[0], dtype=bool)
    
    def siguiente_vecino(self) -> tuple:
        #devuelve el siguiente vecino en el vecindario. Si ya ha sido todos explorados devuelve la misma solucion que generó el vecindario
        siguiente = self.solucion_generadora.copy()

        if(self.permutaciones.shape[1] == 2):
            permutacion = [-1,-1] #permutacion invalida por defecto 
        else:
            permutacion = [-1,-1,-1]

        if(np.any(self.vector_elementos_sin_explorar)):
            indice_elegido = np.argmax(self.vector_elementos_sin_explorar)
            permutacion = self.permutaciones[ self.indice_permutacion[indice_elegido] ][:]
            siguiente[ permutacion[0] ] = 0
            siguiente[ permutacion[1] ] = 1
            if permutacion.shape[0] == 3:
                siguiente[ permutacion[2] ] = 1

            self.vector_elementos_sin_explorar[indice_elegido] = False

        return siguiente, permutacion


#Clase problema

class Problema:
    def __init__(self, matriz_valor, peso_max, vector_pesos) -> None:

        self.matriz_valor = matriz_valor
        self.peso_max = peso_max
        self.vector_pesos = vector_pesos

        self.solucion_actual = Solucion()
        self.solucion_actual.solucion = np.zeros(vector_pesos.shape[0])
        self.solucion_actual.peso = peso_max
        self.solucion_actual.beneficio = 0
    
    #da una solucion inicial aleatoria 
    def solucion_inicial(self) -> np.ndarray:
        # Permuta los indices de manera aleatoria
        indices_aleatorios = np.random.permutation(np.arange(0, self.solucion_actual.solucion.shape[0]))
        for indice in indices_aleatorios:
            if self.vector_pesos[indice] <= self.solucion_actual.peso:
                self.solucion_actual.solucion[indice] = 1
                self.solucion_actual.peso -= self.vector_pesos[indice]

        self.solucion_actual.beneficio = np.sum(self.matriz_valor[self.solucion_actual.solucion.astype(bool), :][:, self.solucion_actual.solucion.astype(bool)])

        return self.solucion_actual

    def factible(self,solucion) -> bool:
        peso_solucion_explorar = np.sum(self.vector_pesos[solucion.astype(bool)])
        if peso_solucion_explorar <= self.peso_max:
            return True
        return False
    
    def calculo_solucion(self, solucion) -> Solucion:
        solucion_calculada = Solucion()
        solucion_calculada.solucion = solucion.copy()
        solucion_calculada.peso = np.sum(self.vector_pesos[solucion.astype(bool)])
        solucion_calculada.beneficio = np.sum(self.matriz_valor[solucion.astype(bool), :][:, solucion.astype(bool)])
        return solucion_calculada

    
    def factorizacion(self, solucion, permutacion) -> Solucion:
        solucion_calculada = Solucion()
        solucion_calculada.solucion = solucion.copy()
        solucion_calculada.peso = np.sum(self.vector_pesos[solucion.astype(bool)])

        solucion_calculada.beneficio = self.solucion_actual.beneficio
        solucion_calculada.beneficio = solucion_calculada.beneficio - np.sum(self.matriz_valor[permutacion[0], :][self.solucion_actual.solucion.astype(bool)]) * 2 + self.matriz_valor[permutacion[0], permutacion[0]]
        solucion_calculada.beneficio = solucion_calculada.beneficio + np.sum(self.matriz_valor[permutacion[1], :][solucion.astype(bool)]) * 2 - self.matriz_valor[permutacion[1], permutacion[1]]
        if permutacion.shape[0] == 3:
             solucion_calculada.beneficio = solucion_calculada.beneficio + np.sum(self.matriz_valor[permutacion[2], :][solucion.astype(bool)]) * 2 - self.matriz_valor[permutacion[2], permutacion[2]] - self.matriz_valor[permutacion[1],permutacion[2]] * 2
        return solucion_calculada

    #toma como la sulucion en la que buscara su entorno la solucion_actual    
    def primer_mejor_vecino(self, N, vecindario = 0) -> bool: #vecindario 0 (vecindario pequeño)
       #en esta busqueda de entorno solo se generan permutaciones de objetos (es decir no aumenta el numero de objetos elegedio en la solucion inicial)
        v = Vecindarios(self.solucion_actual.solucion,vecindario)

        solucion_a_explorar, permutacion = v.siguiente_vecino()

        while (not np.array_equal(solucion_a_explorar, self.solucion_actual.solucion) and N[0] < conf.MAX_EVALUACIONES):

            if self.factible(solucion_a_explorar):
                solucion_a_explorar_calc = self.factorizacion(solucion_a_explorar,permutacion)
                if solucion_a_explorar_calc.beneficio > self.solucion_actual.beneficio:
                    self.solucion_actual = solucion_a_explorar_calc
                    return True
                if not conf.ENTENDER_FACTIVILIDAD_DE_SOLUCION_COMO_PARTE_FUNCION_OBJETIVO:
                    N[0] += 1
            solucion_a_explorar, permutacion = v.siguiente_vecino()

            if conf.ENTENDER_FACTIVILIDAD_DE_SOLUCION_COMO_PARTE_FUNCION_OBJETIVO:
                N[0] += 1
            
        return False
    

def BL_primer_mejor(matriz_valor, peso_max, vector_pesos, vecindario = 0) -> Solucion:
    prob = Problema(matriz_valor, peso_max, vector_pesos)
    prob.solucion_inicial()

    N = [1]
    mejora = prob.primer_mejor_vecino(N)
    

    if conf.VER_GRAFICA_DE_MEJORA_SOLO_PARA_UN_PROBLEMA:
        beneficios = []
        cambios_vecindario = []
        beneficios.append(prob.solucion_actual.beneficio)
        cambios_vecindario.append(False)

    while mejora:
        mejora = prob.primer_mejor_vecino(N)

        if conf.VER_GRAFICA_DE_MEJORA_SOLO_PARA_UN_PROBLEMA and mejora:
            beneficios.append(prob.solucion_actual.beneficio)
            cambios_vecindario.append(False)

        if (not mejora) and vecindario != 0:
            mejora = prob.primer_mejor_vecino(N,vecindario = 1)

            if conf.VER_GRAFICA_DE_MEJORA_SOLO_PARA_UN_PROBLEMA and mejora:
                beneficios.append(prob.solucion_actual.beneficio)
                cambios_vecindario.append(True)
    
    if conf.VER_GRAFICA_DE_MEJORA_SOLO_PARA_UN_PROBLEMA:
        # Graficar los resultados
        for i, beneficio in enumerate(beneficios):
            if cambios_vecindario[i]:
                plt.scatter(i, beneficio, color='red')  # Color distinto para cambios de vecindario
            else:
                plt.scatter(i, beneficio, color='blue')

        plt.xlabel('Iteraciones')
        plt.ylabel('Beneficio')
        plt.title('Evolución del Beneficio en BL')
        plt.show()
    return prob.solucion_actual


# .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------. 
#| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
#| |    ______    | || |  _______     | || |  _________   | || |  _________   | || |  ________    | || |  ____  ____  | |
#| |  .' ___  |   | || | |_   __ \    | || | |_   ___  |  | || | |_   ___  |  | || | |_   ___ `.  | || | |_  _||_  _| | |
#| | / .'   \_|   | || |   | |__) |   | || |   | |_  \_|  | || |   | |_  \_|  | || |   | |   `. \ | || |   \ \  / /   | |
#| | | |    ____  | || |   |  __ /    | || |   |  _|  _   | || |   |  _|  _   | || |   | |    | | | || |    \ \/ /    | |
#| | \ `.___]  _| | || |  _| |  \ \_  | || |  _| |___/ |  | || |  _| |___/ |  | || |  _| |___.' / | || |    _|  |_    | |
#| |  `._____.'   | || | |____| |___| | || | |_________|  | || | |_________|  | || | |________.'  | || |   |______|   | |
#| |              | || |              | || |              | || |              | || |              | || |              | |
#| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
# '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 

def greedy(matriz_valor, peso_max, vector_pesos) -> Solucion:
    #solucion inicial [0, 0, 0, 0, ..., 0]
    solucion = np.zeros(vector_pesos.shape)
    #capacidad inicial peso maximo de la mochila
    capacidad_actual = peso_max

    #calculo de beneficio/peso individual para todos
    beneficio_por_coste = np.diag(matriz_valor) / vector_pesos
#bucle mientras exista algun elemento no asignado de menor o igual peso de la capacidad restante (capacidad_actual)
    while np.any((vector_pesos[~solucion.astype(bool)] <= capacidad_actual)):
        #descarta los elementos que pesan mas que la capacidad actual de la mochila asignandole un beneficio -100
        beneficio_por_coste[vector_pesos > capacidad_actual] = -100

        #mete en la solucion el elemento con mayor beneficio/coste
        indice_ult_intro = np.argmax(beneficio_por_coste) #obtenemos el indice del elemento con mayor beneficio/coste
        beneficio_por_coste[indice_ult_intro] = -101 #asignamos un coste negativo de -101 al elemento que estamos asignando para no volverlo elegir
        capacidad_actual -= vector_pesos[indice_ult_intro] #recalculo la capacidad actual restandole el peso del elemento que estoy asignando
        solucion[indice_ult_intro] = 1 #añado el elemento a la solucion
    
        #recalcular beneficios con elementos ya introducidos
        beneficio_combinado = matriz_valor[indice_ult_intro].copy() #extraigo todos los beneficios combinatorios con el elemento asignado
        beneficio_combinado[indice_ult_intro] = 0 # quito de ese vector el elemento asignado
        beneficio_combinado *= 2
        beneficio_combinado /= vector_pesos # divido todos los beneficios combinatorios con el elemento asignado entre el peso de cada elemento
        beneficio_por_coste[beneficio_por_coste >= 0] += beneficio_combinado[beneficio_por_coste >= 0] 
        #incremento estos beneficios para los elementontos con beneficios aun positivos es decir los que no sobrepasan la capacidad (por ahora) y a los que no estan asignados
    prob = Problema(matriz_valor, peso_max, vector_pesos)
    solucion = prob.calculo_solucion(solucion)
    return solucion
