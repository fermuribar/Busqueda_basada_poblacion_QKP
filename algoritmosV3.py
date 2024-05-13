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

        densidades = np.sum(matriz_valor, axis=1) / vector_pesos #antes era beneficio unitario 
        self.indices_por_densidad = np.argsort(densidades)[::-1]
    
    #da una solucion inicial aleatoria 
    def solucion_inicial(self) -> Solucion:
        # Permuta los indices de manera aleatoria
        indices_aleatorios = np.random.permutation(np.arange(0, self.solucion_actual.solucion.shape[0]))
        for indice in indices_aleatorios:
            if self.vector_pesos[indice] <= self.solucion_actual.peso:
                self.solucion_actual.solucion[indice] = 1
                self.solucion_actual.peso -= self.vector_pesos[indice]

        self.solucion_actual.beneficio = np.sum(self.matriz_valor[self.solucion_actual.solucion.astype(bool), :][:, self.solucion_actual.solucion.astype(bool)])

        return self.solucion_actual

    #comprueba si una solucion es factible
    def factible(self,solucion) -> bool:
        peso_solucion_explorar = np.sum(self.vector_pesos[solucion.astype(bool)])
        if peso_solucion_explorar <= self.peso_max:
            return True
        return False
    
    #metodo para rellenar la mochila con elementos que aun quepan en ella
    def completar(self, solucion) -> np.array:
        indices_0 = np.where(solucion == 0)[0]
        aleatorios = np.random.permutation(indices_0)
        peso_dispo = self.peso_max - np.sum(self.vector_pesos[solucion.astype(bool)])
        i = 0
        while i < len(aleatorios) and np.any(self.vector_pesos[aleatorios] <= peso_dispo):
            if self.vector_pesos[aleatorios[i]] <= peso_dispo:
                 solucion[aleatorios[i]] = 1
                 peso_dispo -= self.vector_pesos[aleatorios[i]]
            i+=1

        return solucion
    
    #metodo para hacer que una solucion no factible se haga factible (aleatoriamente)
        #al final se llama a completar
    def factibilizar(self, solucion) -> np.array:
        indices_1 = np.where(solucion == 1)[0]
        aleatorios = np.random.permutation(indices_1)
        for indice in aleatorios:
            solucion[indice] = 0
            if self.factible(solucion):
                break
        solucion = self.completar(solucion)
        return solucion
    
    #calcula todos los parametros de la estructura soluciuon
    def calculo_solucion(self, solucion) -> Solucion:
        solucion_calculada = Solucion()
        solucion_calculada.solucion = solucion.copy()
        solucion_calculada.peso = np.sum(self.vector_pesos[solucion.astype(bool)])
        solucion_calculada.beneficio = np.sum(self.matriz_valor[solucion.astype(bool), :][:, solucion.astype(bool)])
        return solucion_calculada
    
    #calcula los parametros "factorizando" de una solucion a partir de una anterior
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

    #obtencion del primer mejor vecino de un vecindario (BL)
        #toma como la sulucion en la que buscara su entorno la solucion_actual    
    def primer_mejor_vecino(self, N, vecindario = 0, lim = 0) -> bool: #vecindario 0 (vecindario pequeño)
       #en esta busqueda de entorno solo se generan permutaciones de objetos (es decir no aumenta el numero de objetos elegedio en la solucion inicial)
        v = Vecindarios(self.solucion_actual.solucion,vecindario)

        solucion_a_explorar, permutacion = v.siguiente_vecino()

        if lim == 0:
            limite = conf.MAX_EVALUACIONES
        else:
            limite = lim

        while (not np.array_equal(solucion_a_explorar, self.solucion_actual.solucion) and N[0] < limite):

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
    
    #crea una poblacion de soluciones aleatorias
    def poblacion_inicial(self) -> list:
        pop = []
        for i in range(0,conf.POBLACION):
            self.solucion_actual = Solucion()
            self.solucion_actual.solucion = np.zeros(self.vector_pesos.shape[0])
            self.solucion_actual.peso = self.peso_max
            self.solucion_actual.beneficio = 0
            pop.append(self.solucion_inicial())
        return pop
    
    #operador de cure en dos puntos
    def cruce_intercambio_puntos(self, padre1, padre2) -> tuple:
        indices_cruce = np.random.randint(0,padre1.shape[0], size=2)
        hijo1 = padre1.copy()
        hijo2 = padre2.copy()
        
        hijo1[indices_cruce.min():indices_cruce.max() + 1] = padre2[indices_cruce.min():indices_cruce.max() + 1].copy()
        hijo2[indices_cruce.min():indices_cruce.max() + 1] = padre1[indices_cruce.min():indices_cruce.max() + 1].copy()

        if not self.factible(hijo1):
            hijo1 = self.factibilizar(hijo1)

        if not self.factible(hijo2):
            hijo2 = self.factibilizar(hijo2)

        hijo1 = self.calculo_solucion(hijo1)
        hijo2 = self.calculo_solucion(hijo2)
        return (hijo1, hijo2)
    
    #operador propuesto: se copian los padres (hijo1 = padre1) y se intentan añadir mas objetos del otro padre
        #con prioridad de beneficio/peso. POCO EFICIENTE; hipotesis: como las soluciones de partidas ya eran factibles 
        #y "completas" no se copian del padre2 nada. Es decir las soluciones no se diferencian muchos de los padres.
    def cruce_propuesto3(self, padre1, padre2) -> tuple:
        hijo1 = padre1.copy()
        hijo2 = padre2.copy()

        for i in self.indices_por_densidad:
            if padre2[i] == 1 and padre1[i] == 0:
                hijo1[i] = 1
                if not self.factible(hijo1):
                    hijo1[i] = 0
            if padre1[i] == 1 and padre2[i] == 0:
                hijo2[i] = 1
                if not self.factible(hijo2):
                    hijo2[i] = 0


        hijo1 = self.completar(hijo1)
        hijo2 = self.completar(hijo2)

        hijo1 = self.calculo_solucion(hijo1)
        hijo2 = self.calculo_solucion(hijo2)
        return (hijo1, hijo2)
    
    #operador propuesto: es igual que el curuce en dos puntos pero en el momento de factivilizar la soluciones obtenidas 
        #y completar la solucion se usa informacion del problema con el beneficio/peso. ultimo en term
    def cruce_propuesto1(self, padre1, padre2) -> tuple:
        indices_cruce = np.random.randint(0,padre1.shape[0], size=2)
        hijo1 = padre1.copy()
        hijo2 = padre2.copy()
        
        hijo1[indices_cruce.min():indices_cruce.max() + 1] = padre2[indices_cruce.min():indices_cruce.max() + 1].copy()
        hijo2[indices_cruce.min():indices_cruce.max() + 1] = padre1[indices_cruce.min():indices_cruce.max() + 1].copy()

        if not self.factible(hijo1):
            for indice in self.indices_por_densidad[::-1]:
                if hijo1[indice] == 1:
                    hijo1[indice] = 0
                    if self.factible(hijo1):
                        break

        peso_dispo = self.peso_max - np.sum(self.vector_pesos[hijo1.astype(bool)])
        for indice in self.indices_por_densidad:
            if hijo1[indice] == 0 and self.vector_pesos[indice] <= peso_dispo:
                 hijo1[indice] = 1
                 peso_dispo -= self.vector_pesos[indice]

        if not self.factible(hijo2):
            for indice in self.indices_por_densidad[::-1]:
                if hijo2[indice] == 1:
                    hijo2[indice] = 0
                    if self.factible(hijo2):
                        break
        
        peso_dispo = self.peso_max - np.sum(self.vector_pesos[hijo2.astype(bool)])
        for indice in self.indices_por_densidad:
            if hijo2[indice] == 0 and self.vector_pesos[indice] <= peso_dispo:
                 hijo2[indice] = 1
                 peso_dispo -= self.vector_pesos[indice]

        hijo1 = self.calculo_solucion(hijo1)
        hijo2 = self.calculo_solucion(hijo2)
        return (hijo1, hijo2)
    
    def cruce_propuesto2(self, padre1, padre2) -> tuple:
        hijo1 = np.logical_and(padre1, padre2)
        hijo2 = np.logical_xor(padre1, padre2)


        if not self.factible(hijo1):
            for indice in self.indices_por_densidad[::-1]:
                if hijo1[indice] == 1:
                    hijo1[indice] = 0
                    if self.factible(hijo1):
                        break

        peso_dispo = self.peso_max - np.sum(self.vector_pesos[hijo1.astype(bool)])
        for indice in self.indices_por_densidad:
            if hijo1[indice] == 0 and self.vector_pesos[indice] <= peso_dispo:
                 hijo1[indice] = 1
                 peso_dispo -= self.vector_pesos[indice]

        if not self.factible(hijo2):
            for indice in self.indices_por_densidad[::-1]:
                if hijo2[indice] == 1:
                    hijo2[indice] = 0
                    if self.factible(hijo2):
                        break
        
        peso_dispo = self.peso_max - np.sum(self.vector_pesos[hijo2.astype(bool)])
        for indice in self.indices_por_densidad:
            if hijo2[indice] == 0 and self.vector_pesos[indice] <= peso_dispo:
                 hijo2[indice] = 1
                 peso_dispo -= self.vector_pesos[indice]

        hijo1 = self.calculo_solucion(hijo1)
        hijo2 = self.calculo_solucion(hijo2)
        return (hijo1, hijo2)
    
    #operador de mutacion
    def mutacion(self, sol) -> Solucion:
        mutada = sol.copy()
        m = 0
        while m < sol.shape[0]*conf.PROBABILIDAD_MUTACION:
            indices_mutado = np.random.randint(0, sol.shape[0])
            mutada[indices_mutado] = 1 if mutada[indices_mutado] == 0 else 0
            if self.factible(mutada):
                m += 1
            else:
                mutada[indices_mutado] = 1 if mutada[indices_mutado] == 0 else 0

        mutada = self.completar(mutada)
        
        return self.calculo_solucion(mutada)

#   ____  _      
#  |  _ \| |     
#  | |_) | |     
#  |  _ <| |     
#  | |_) | |____ 
#  |____/|______|
    
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
        if vecindario == 1:
            plt.title('Evolución del Beneficio en BL+')
        else:
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

#            _____  _____ 
#      /\   / ____|/ ____|
#     /  \ | |  __| |  __ 
#    / /\ \| | |_ | | |_ |
#   / ____ \ |__| | |__| |
#  /_/    \_\_____|\_____|

def BL_primer_mejor_meme(matriz_valor, peso_max, vector_pesos, solucion, N, lim) -> Solucion:
    prob = Problema(matriz_valor, peso_max, vector_pesos)

    prob.solucion_actual.solucion = solucion.solucion.copy()
    prob.solucion_actual.beneficio = solucion.beneficio
    prob.solucion_actual.peso = solucion.peso

    mejora = prob.primer_mejor_vecino(N, lim = lim)
    while mejora:
        mejora = prob.primer_mejor_vecino(N, lim = lim)

    return prob.solucion_actual


def torneo_de_tres(pop):
    #indices random
    pos1 = np.random.randint(0, conf.POBLACION)
    pos2 = np.random.randint(0, conf.POBLACION)
    pos3 = np.random.randint(0, conf.POBLACION)

    #asegura que son distintos
    while pos2 == pos1 or pos2 == pos3:
        pos2 = np.random.randint(0, conf.POBLACION)

    while pos3 == pos1 or pos3 == pos2:
        pos3 = np.random.randint(0, conf.POBLACION)

    if pop[pos1].beneficio < pop[pos2].beneficio and pop[pos3].beneficio < pop[pos2].beneficio:
        return pop[pos2]
    elif pop[pos1].beneficio < pop[pos3].beneficio and pop[pos2].beneficio < pop[pos3].beneficio:
        return pop[pos3]
    else:
        return pop[pos1]
    
def mejor_de_pop(pop):
    beneficios = np.array([indi.beneficio for indi in pop])
    ind_orden = np.argsort(beneficios)[::-1]
    return pop[ind_orden[0]]

def peor_de_pop(pop):       #no se usa
    beneficios = np.array([indi.beneficio for indi in pop])
    ind_orden = np.argsort(beneficios)
    return pop[ind_orden[0]]

def agg(matriz_valor, peso_max, vector_pesos, cruce = 0, meme = 0) -> Solucion:
    p = Problema(matriz_valor, peso_max, vector_pesos)
    pop = p.poblacion_inicial()
    beneficios = np.array([indi.beneficio for indi in pop])
    newpop = pop.copy()
    evaluadas = len(pop)
    eva_bl = 0
    
    generacion = 1

    mejor = pop[np.argmax(beneficios)]
    if conf.VER_GRAFICA_DE_MEJORA_SOLO_PARA_UN_PROBLEMA:
        historial_mejor = []
        historial_mejor.append(mejor.beneficio)
        historial_media = []
        historial_media.append(beneficios.mean())

    while evaluadas < conf.MAX_EVALUACIONES:
        #seleccion
        for i in newpop:
            #Copia el ganador del torneo
            ganador = torneo_de_tres(pop)
            i = ganador

        #numero de cruces
        total_cruces = conf.POBLACION*conf.PROBABILIDAD_CRUZE

        i = 0
        while i < total_cruces:
            padre1 = pop[i]
            padre2 = pop[i+1]
            
            if cruce == 0:
                h1, h2 = p.cruce_intercambio_puntos(padre1.solucion,padre2.solucion)
            elif cruce == 1:
                h1, h2 = p.cruce_propuesto1(padre1.solucion,padre2.solucion)

            newpop[i] = h1
            newpop[i+1] = h2
            i += 2
            evaluadas += 2 

        #mutacion
        total_mutar = conf.POBLACION*conf.PROBABILIDAD_MUTACION

        for i in range(0, int(total_mutar)):
            posi = np.random.randint(0, conf.POBLACION)
            newpop[posi] = p.mutacion(pop[posi].solucion)
            if posi >= total_cruces: # para no contar los cruces
                evaluadas += 1


        if meme != 0 and generacion % 10 == 0:  #mover antes de incluso el elitismo-----------________________------------___________
            if meme == 1:
                for cromosoma in newpop:
                    if evaluadas >= conf.MAX_EVALUACIONES:
                        break
                    elif conf.MAX_EVALUACIONES - evaluadas < cromosoma.solucion.shape[0]:
                        lim = conf.MAX_EVALUACIONES - evaluadas
                    else:
                        lim = cromosoma.solucion.shape[0]

                    N = [1]
                    cromosoma = BL_primer_mejor_meme(matriz_valor, peso_max, vector_pesos, cromosoma, N, lim)
                    evaluadas += N[0]
                    eva_bl += N[0]
            elif meme == 2:
                cromosomas_BL = conf.POBLACION * conf.PROBABILIDAD_MEME2
                for i in range(0, int(cromosomas_BL)):
                    cromosoma = newpop[np.random.randint(0, conf.POBLACION)]
                    if evaluadas >= conf.MAX_EVALUACIONES:
                        break
                    elif conf.MAX_EVALUACIONES - evaluadas < cromosoma.solucion.shape[0]:
                        lim = conf.MAX_EVALUACIONES - evaluadas
                    else:
                        lim = cromosoma.solucion.shape[0]
    
                    N = [1]
                    cromosoma = BL_primer_mejor_meme(matriz_valor, peso_max, vector_pesos, cromosoma, N, lim)
                    evaluadas += N[0]
                    eva_bl += N[0]
            elif meme == 3:
                newpop = sorted(newpop, key=lambda x : x.beneficio)[::-1]
                cromosomas_BL = conf.POBLACION * conf.PROBABILIDAD_MEME2
                for i in range(0, int(cromosomas_BL)):
                    if evaluadas >= conf.MAX_EVALUACIONES:
                        break
                    elif conf.MAX_EVALUACIONES - evaluadas < newpop[0].solucion.shape[0]:
                        lim = conf.MAX_EVALUACIONES - evaluadas
                    else:
                        lim = newpop[0].solucion.shape[0]

                    N = [1]
                    newpop[i] = BL_primer_mejor_meme(matriz_valor, peso_max, vector_pesos, newpop[i], N, lim)
                    evaluadas += N[0]
                    eva_bl += N[0]

        beneficios = np.array([indi.beneficio for indi in newpop]) #mejorable

        #rescata mejor
        mejor_new = newpop[np.argmax(beneficios)]
        
        if mejor.beneficio > mejor_new.beneficio:
            peor = newpop[np.argmin(beneficios)]
            peor.solucion = mejor.solucion.copy()
            peor.beneficio = mejor.beneficio
        else:
            mejor = mejor_new

        if conf.VER_GRAFICA_DE_MEJORA_SOLO_PARA_UN_PROBLEMA:
            historial_mejor.append(mejor.beneficio)
            historial_media.append(beneficios.mean())

        #Remplazo
        pop = newpop.copy()

        generacion +=1

    if conf.VER_GRAFICA_DE_MEJORA_SOLO_PARA_UN_PROBLEMA:
        # Graficar los resultados
        plt.plot(historial_mejor)
        plt.plot(historial_media)
        plt.xlabel('Iteraciones')
        plt.ylabel('Beneficio')
        if cruce == 1:
            plt.title('Evolución del Beneficio en AGG 1')
        elif meme == 1:
            plt.title('Evolución del Beneficio en AM1')
        elif meme == 2:
            plt.title('Evolución del Beneficio en AM2')
        elif meme == 3:
            plt.title('Evolución del Beneficio en AM3')
        else:
            plt.title('Evolución del Beneficio en AGG')
        plt.show()

    return (mejor_de_pop(pop), evaluadas, eva_bl)

#            _____ ______ 
#      /\   / ____|  ____|
#     /  \ | |  __| |__   
#    / /\ \| | |_ |  __|  
#   / ____ \ |__| | |____ 
#  /_/    \_\_____|______|

def ramplazar_peores(pop, h1, h2):
    beneficios = np.array([indi.beneficio for indi in pop])
    ind_orden = np.argsort(beneficios)
    if h1.beneficio > h2.beneficio:
        if h1.beneficio > pop[ind_orden[2]].beneficio:
            pop[ind_orden[1]] = h1
            if h2.beneficio > pop[ind_orden[1]].beneficio:
                pop[ind_orden[2]] = h2
        else:
            if h1.beneficio > pop[-1].beneficio:
                pop[ind_orden[2]] = h1
    else:
        if h2.beneficio > pop[ind_orden[2]].beneficio:
            pop[ind_orden[2]] = h2
            if h1.beneficio > pop[ind_orden[1]].beneficio:
                pop[ind_orden[1]] = h1
        else:
            if h2.beneficio > pop[ind_orden[1]].beneficio:
                pop[ind_orden[1]] = h2




def age(matriz_valor, peso_max, vector_pesos, cruce = 0) -> Solucion:
    p = Problema(matriz_valor, peso_max, vector_pesos)
    pop = p.poblacion_inicial()
    evaluadas = conf.POBLACION
    
    if conf.VER_GRAFICA_DE_MEJORA_SOLO_PARA_UN_PROBLEMA:
        historial = []
        historial.append(mejor_de_pop(pop).beneficio)
        historial_media = []
        beneficios = np.array([indi.beneficio for indi in pop])
        historial_media.append(beneficios.mean())

    while evaluadas < conf.MAX_EVALUACIONES:
        #Seleccion por torneo
        padre1 = torneo_de_tres(pop)
        padre2 = torneo_de_tres(pop)

        #cruce
        if cruce == 0:
            h1, h2 = p.cruce_intercambio_puntos(padre1.solucion,padre2.solucion)
        elif cruce == 1:
            h1, h2 = p.cruce_propuesto1(padre1.solucion,padre2.solucion)

        #mutacion
        if conf.PROBABILIDAD_MUTACION >= np.random.rand():
            h1 = p.mutacion(h1.solucion)

        if conf.PROBABILIDAD_MUTACION >= np.random.rand():
            h2 = p.mutacion(h2.solucion)
        
        evaluadas += 2

        #Remplaza los dos peores
        ramplazar_peores(pop, h1, h2)
        if conf.VER_GRAFICA_DE_MEJORA_SOLO_PARA_UN_PROBLEMA:
            historial.append(mejor_de_pop(pop).beneficio)
            beneficios = np.array([indi.beneficio for indi in pop])
            historial_media.append(beneficios.mean())

    if conf.VER_GRAFICA_DE_MEJORA_SOLO_PARA_UN_PROBLEMA:
        # Graficar los resultados
        plt.plot(historial)
        plt.plot(historial_media)
        plt.xlabel('Iteraciones')
        plt.ylabel('Beneficio')
        if cruce == 1:
            plt.title('Evolución del Beneficio en AGE 1')
        else:
            plt.title('Evolución del Beneficio en AGE')
        plt.show()

    return (mejor_de_pop(pop), evaluadas)

