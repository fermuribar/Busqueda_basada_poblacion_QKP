import config as conf
import algoritmosV3 as alg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

#  __  __          _____ _   _ 
# |  \/  |   /\   |_   _| \ | |
# | \  / |  /  \    | | |  \| |
# | |\/| | / /\ \   | | | . ` |
# | |  | |/ ____ \ _| |_| |\  |
# |_|  |_/_/    \_\_____|_| \_|

def main():
    inicio_global = time.time()
    if conf.VER_GRAFICA_DE_MEJORA_SOLO_PARA_UN_PROBLEMA:
        ruta_archivo = conf.PROBLEMA_PARA_VER_GRAFICA # Reemplaza con la ruta de tu archivo
        if not os.path.isfile(ruta_archivo):
            print('Fichero: {}      NO ENCONTRADO'.format(conf.PROBLEMA_PARA_VER_GRAFICA))
            return
        matriz_valores, peso_maximo, vector_pesos = alg.procesar_archivo(ruta_archivo)
        np.random.seed(conf.SEMILLA)
        solucion = alg.BL_primer_mejor(matriz_valores,peso_maximo,vector_pesos)
        print("BL -> ({}) Con un peso disponible de: {}    y bondad total de: {}".format(ruta_archivo, peso_maximo - solucion.peso ,solucion.beneficio))
        print('')
        np.random.seed(conf.SEMILLA)
        solucion_mejora = alg.BL_primer_mejor(matriz_valores,peso_maximo,vector_pesos, vecindario = 1)
        print("BL+ -> ({}) Con un peso disponible de: {}    y bondad total de: {}".format(ruta_archivo, peso_maximo - solucion_mejora.peso ,solucion_mejora.beneficio))
        print('')
        np.random.seed(conf.SEMILLA)
        sol_agg, eva_agg, _ = alg.agg(matriz_valores, peso_maximo, vector_pesos)
        print("Agg -> ({}) Con un peso disponible de: {}    y bondad total de: {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_agg.peso ,sol_agg.beneficio, eva_agg))
        print('')
        np.random.seed(conf.SEMILLA)
        sol_agg_1, eva_agg_1, _ = alg.agg(matriz_valores, peso_maximo, vector_pesos,cruce=1)
        print("Agg_1 -> ({}) Con un peso disponible de: {}    y bondad total de: {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_agg_1.peso ,sol_agg_1.beneficio, eva_agg_1))
        print('')
        np.random.seed(conf.SEMILLA)
        sol_age, eva_age = alg.age(matriz_valores, peso_maximo, vector_pesos)
        print("Age -> ({}) Con un peso disponible de: {}    y bondad total de: {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_age.peso ,sol_age.beneficio, eva_age))
        print('')
        np.random.seed(conf.SEMILLA)
        sol_age_1, eva_age_1 = alg.age(matriz_valores, peso_maximo, vector_pesos,cruce=1)
        print("Age_1 -> ({}) Con un peso disponible de: {}    y bondad total de: {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_age_1.peso ,sol_age_1.beneficio, eva_age_1))
        print('')
        np.random.seed(conf.SEMILLA)
        sol_am1, eva_am1, bl_am1 = alg.agg(matriz_valores, peso_maximo, vector_pesos, meme=1)
        print("Am1 -> ({}) Con un peso disponible de: {}    y bondad total de: {} || evaluaciones P: {};  evaluaciones bl: {}".format(ruta_archivo, peso_maximo - sol_am1.peso ,sol_am1.beneficio, eva_am1-bl_am1, bl_am1))
        print('')
        np.random.seed(conf.SEMILLA)
        sol_am2, eva_am2, bl_am2 = alg.agg(matriz_valores, peso_maximo, vector_pesos, meme=2)
        print("Am2 -> ({}) Con un peso disponible de: {}    y bondad total de: {} || evaluaciones P: {};  evaluaciones bl: {}".format(ruta_archivo, peso_maximo - sol_am2.peso ,sol_am2.beneficio, eva_am2-bl_am2, bl_am2))
        print('')
        np.random.seed(conf.SEMILLA)
        sol_am3, eva_am3, bl_am3 = alg.agg(matriz_valores, peso_maximo, vector_pesos, meme=3)
        print("Am3 -> ({}) Con un peso disponible de: {}    y bondad total de: {} || evaluaciones P: {};  evaluaciones bl: {}".format(ruta_archivo, peso_maximo - sol_am3.peso ,sol_am3.beneficio, eva_am3-bl_am3, bl_am3))
        print('')
        return

    # Crear un DataFrame vacío con las columnas específicas
    if conf.MEJORA_BL:
        columnas = ['Nombre del problema',  'Solucion BL', 'Peso BL', 'Bondad BL', 'Tiempo BL', 
                                            'Solucion Greedy', 'Peso Greedy', 'Bondad Greedy', 'Tiempo Greedy', 
                                            'Solucion agg', 'Peso agg', 'Bondad agg', 'Tiempo agg', 
                                            'Solucion agg_1', 'Peso agg_1', 'Bondad agg_1', 'Tiempo agg_1', 
                                            'Solucion age', 'Peso age', 'Bondad age', 'Tiempo age', 
                                            'Solucion age_1', 'Peso age_1', 'Bondad age_1', 'Tiempo age_1', 
                                            'Solucion am1', 'Peso am1', 'Bondad am1', 'Tiempo am1',
                                            'Solucion am2', 'Peso am2', 'Bondad am2', 'Tiempo am2',
                                            'Solucion am3', 'Peso am3', 'Bondad am3', 'Tiempo am3',
                                            'Solucion BL+', 'Peso BL+', 'Bondad BL+', 'Tiempo BL+']
    else:
        columnas = ['Nombre del problema',  'Solucion BL', 'Peso BL', 'Bondad BL', 'Tiempo BL', 
                                            'Solucion Greedy', 'Peso Greedy', 'Bondad Greedy', 'Tiempo Greedy', 
                                            'Solucion agg', 'Peso agg', 'Bondad agg', 'Tiempo agg', 
                                            'Solucion agg_1', 'Peso agg_1', 'Bondad agg_1', 'Tiempo agg_1',
                                            'Solucion age', 'Peso age', 'Bondad age', 'Tiempo age',
                                            'Solucion age_1', 'Peso age_1', 'Bondad age_1', 'Tiempo age_1', 
                                            'Solucion am1', 'Peso am1', 'Bondad am1', 'Tiempo am1',
                                            'Solucion am2', 'Peso am2', 'Bondad am2', 'Tiempo am2',
                                            'Solucion am3', 'Peso am3', 'Bondad am3', 'Tiempo am3']
        
    tabla = pd.DataFrame(columns=columnas)
    # Uso de la función
    Carga=0
    if (conf.SOLO_EJECUTAR_100 and conf.SOLO_EJECUTAR_200) or (conf.SOLO_EJECUTAR_100 and conf.SOLO_EJECUTAR_300) or (conf.SOLO_EJECUTAR_200 and conf.SOLO_EJECUTAR_300):
        print("Error mala configuracion en los parametros SOLO_EJECUTAR_X. Mas de uno activado")
        return
    elif conf.SOLO_EJECUTAR_100:
        k = 1
        p = 2
    elif conf.SOLO_EJECUTAR_200:
        k = 2
        p = 3
    elif conf.SOLO_EJECUTAR_300:
        k = 3
        p = 4
    else:
        k = 1
        p = 4
    for c in range(k,p):
        for b in range(1,5):
            for i in range(1,11):
                ruta_archivo = "data/jeu_{}_{}_{}.txt".format(c * 100,b * 25,i)  # Reemplaza con la ruta de tu archivo
                if not os.path.isfile(ruta_archivo):
                    continue
                if not conf.MOSTRAR_CADA_SALIDA:
                    os.system(conf.CLEAR)
                    if Carga == 0:
                        print('->')
                        Carga=1
                    elif Carga == 1:
                        print('-->')
                        Carga=2
                    elif Carga == 2:
                        print('---->')
                        Carga=0

                matriz_valores, peso_maximo, vector_pesos = alg.procesar_archivo(ruta_archivo)

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                solucion = alg.BL_primer_mejor(matriz_valores,peso_maximo,vector_pesos)
                fin = time.time()
                duracion_BL = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(solucion.solucion.astype(int))
                    print("BL -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {}".format(ruta_archivo, peso_maximo - solucion.peso ,solucion.beneficio, duracion_BL))
                    print('')

                inicio = time.time()
                sol = alg.greedy(matriz_valores, peso_maximo, vector_pesos)
                fin = time.time()
                duracion_Greedy = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol.solucion.astype(int))
                    print("Greedy -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {}".format(ruta_archivo, peso_maximo - sol.peso ,sol.beneficio, duracion_Greedy))
                    print('')

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                sol_agg, eva_agg, _ = alg.agg(matriz_valores, peso_maximo, vector_pesos)
                fin = time.time()
                duracion_agg = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol_agg.solucion.astype(int))
                    print("Agg -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_agg.peso ,sol_agg.beneficio, duracion_agg, eva_agg))
                    print('')

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                sol_agg_1, eva_agg_1, _ = alg.agg(matriz_valores, peso_maximo, vector_pesos,cruce=1)
                fin = time.time()
                duracion_agg_1 = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol_agg_1.solucion.astype(int))
                    print("Agg_1 -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_agg_1.peso ,sol_agg_1.beneficio, duracion_agg_1, eva_agg_1))
                    print('')

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                sol_age, eva_age = alg.age(matriz_valores, peso_maximo, vector_pesos)
                fin = time.time()
                duracion_age = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol_age.solucion.astype(int))
                    print("Age -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_age.peso ,sol_age.beneficio, duracion_age, eva_age))
                    print('')

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                sol_age_1, eva_age_1 = alg.age(matriz_valores, peso_maximo, vector_pesos,cruce=1)
                fin = time.time()
                duracion_age_1 = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol_age_1.solucion.astype(int))
                    print("Age_1 -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones: {}".format(ruta_archivo, peso_maximo - sol_age_1.peso ,sol_age_1.beneficio, duracion_age_1, eva_age_1))
                    print('')

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                sol_am1, eva_am1, bl_am1 = alg.agg(matriz_valores, peso_maximo, vector_pesos, meme=1)
                fin = time.time()
                duracion_am1 = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol_am1.solucion.astype(int))
                    print("Am1 -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones P: {};  evaluaciones bl: {}".format(ruta_archivo, peso_maximo - sol_am1.peso ,sol_am1.beneficio, duracion_am1, eva_am1-bl_am1, bl_am1))
                    print('')

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                sol_am2, eva_am2, bl_am2 = alg.agg(matriz_valores, peso_maximo, vector_pesos, meme=2)
                fin = time.time()
                duracion_am2 = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol_am2.solucion.astype(int))
                    print("Am2 -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones P: {};  evaluaciones bl: {}".format(ruta_archivo, peso_maximo - sol_am2.peso ,sol_am2.beneficio, duracion_am2, eva_am2-bl_am2, bl_am2))
                    print('')

                np.random.seed(conf.SEMILLA)
                inicio = time.time()
                sol_am3, eva_am3, bl_am3 = alg.agg(matriz_valores, peso_maximo, vector_pesos, meme=3)
                fin = time.time()
                duracion_am3 = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol_am3.solucion.astype(int))
                    print("Am3 -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {} || evaluaciones P: {};  evaluaciones bl: {}".format(ruta_archivo, peso_maximo - sol_am3.peso ,sol_am3.beneficio, duracion_am3, eva_am3-bl_am3, bl_am3))
                    print('')

                if conf.MEJORA_BL:
                    np.random.seed(conf.SEMILLA)
                    inicio = time.time()
                    solucion_mejora = alg.BL_primer_mejor(matriz_valores,peso_maximo,vector_pesos, vecindario = 1)
                    fin = time.time()
                    duracion_BL_mejora = fin - inicio
                    if conf.MOSTRAR_CADA_SALIDA:
                        print(solucion_mejora.solucion.astype(int))
                        print("BL+ -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {}".format(ruta_archivo, peso_maximo - solucion_mejora.peso ,solucion_mejora.beneficio, duracion_BL_mejora))
                        print('')
                        
                    tabla.loc[len(tabla)] = [ruta_archivo,  solucion.solucion, solucion.peso, solucion.beneficio, duracion_BL, 
                                                            sol.solucion, sol.peso, sol.beneficio, duracion_Greedy, 
                                                            sol_agg.solucion, sol_agg.peso, sol_agg.beneficio, duracion_agg, 
                                                            sol_agg_1.solucion, sol_agg_1.peso, sol_agg_1.beneficio, duracion_agg_1, 
                                                            sol_age.solucion, sol_age.peso, sol_age.beneficio, duracion_age,
                                                            sol_age_1.solucion, sol_age_1.peso, sol_age_1.beneficio, duracion_age_1,
                                                            sol_am1.solucion, sol_am1.peso, sol_am1.beneficio, duracion_am1, 
                                                            sol_am2.solucion, sol_am2.peso, sol_am2.beneficio, duracion_am2, 
                                                            sol_am3.solucion, sol_am3.peso, sol_am3.beneficio, duracion_am3, 
                                                            solucion_mejora.solucion, solucion_mejora.peso, solucion_mejora.beneficio, duracion_BL_mejora]
                else:
                    tabla.loc[len(tabla)] = [ruta_archivo,  solucion.solucion, solucion.peso, solucion.beneficio, duracion_BL, 
                                                            sol.solucion, sol.peso, sol.beneficio, duracion_Greedy, 
                                                            sol_agg.solucion, sol_agg.peso, sol_agg.beneficio, duracion_agg, 
                                                            sol_agg_1.solucion, sol_agg_1.peso, sol_agg_1.beneficio, duracion_agg_1, 
                                                            sol_age.solucion, sol_age.peso, sol_age.beneficio, duracion_age,
                                                            sol_age_1.solucion, sol_age_1.peso, sol_age_1.beneficio, duracion_age_1,
                                                            sol_am1.solucion, sol_am1.peso, sol_am1.beneficio, duracion_am1, 
                                                            sol_am2.solucion, sol_am2.peso, sol_am2.beneficio, duracion_am2, 
                                                            sol_am3.solucion, sol_am3.peso, sol_am3.beneficio, duracion_am3]
                print('')
    
    if not conf.MOSTRAR_CADA_SALIDA:
        os.system(conf.CLEAR)
        print('-----------------------------------FIN----------------------------------------')

    fin_global = time.time()
    duracion_global = fin_global - inicio_global

    print("todo las ejecucuciones han tardado: {}".format(duracion_global))


#  _______    _     _              _____                                           _             
# |__   __|  | |   | |            / ____|                                         (_)            
#    | | __ _| |__ | | __ _ ___  | |     ___  _ __ ___  _ __   __ _ _ __ __ _  ___ _  ___  _ __  
#    | |/ _` | '_ \| |/ _` / __| | |    / _ \| '_ ` _ \| '_ \ / _` | '__/ _` |/ __| |/ _ \| '_ \ 
#    | | (_| | |_) | | (_| \__ \ | |___| (_) | | | | | | |_) | (_| | | | (_| | (__| | (_) | | | |
#    |_|\__,_|_.__/|_|\__,_|___/  \_____\___/|_| |_| |_| .__/ \__,_|_|  \__,_|\___|_|\___/|_| |_|
#                                                      | |                                       
#                                                      |_|                                       

    # Paso 1: Extraer 'Tamaño' y 'Densidad' del nombre del problema utilizando expresiones regulares
    # El primer grupo captura secuencias de dígitos (\d+) que siguen al patrón "jeu_" y antes de "_"
    # El segundo grupo captura secuencias de dígitos (\d+) que están entre dos "_"
    tabla['Tamaño'] = tabla['Nombre del problema'].str.extract(r'jeu_(\d+)_').astype(int)
    tabla['Densidad'] = tabla['Nombre del problema'].str.extract(r'jeu_\d+_(\d+)_').astype(int)

    # Paso 2: Agrupar los datos por 'Tamaño' y 'Densidad' y calcular la media para las métricas 'Bondad BL' y 'Tiempo BL'
    resultados_BL = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad BL', 'Tiempo BL']].mean().reset_index()

    # Paso 3: Agrupar los datos por 'Tamaño' y 'Densidad' y calcular la media para las métricas 'Bondad Greedy' y 'Tiempo Greedy'
    resultados_Greedy = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad Greedy', 'Tiempo Greedy']].mean().reset_index()

    # Redondear la bondad a dos decimales para el DataFrame de BL
    resultados_BL['Bondad BL'] = resultados_BL['Bondad BL'].round(2)

    # Redondear la bondad a dos decimales para el DataFrame de Greedy
    resultados_Greedy['Bondad Greedy'] = resultados_Greedy['Bondad Greedy'].round(2)

    resultados_agg = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad agg', 'Tiempo agg']].mean().reset_index()
    resultados_agg['Bondad agg'] = resultados_agg['Bondad agg'].round(2)

    resultados_agg_1 = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad agg_1', 'Tiempo agg_1']].mean().reset_index()
    resultados_agg_1['Bondad agg_1'] = resultados_agg_1['Bondad agg_1'].round(2)

    resultados_age = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad age', 'Tiempo age']].mean().reset_index()
    resultados_age['Bondad age'] = resultados_age['Bondad age'].round(2)

    resultados_age_1 = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad age_1', 'Tiempo age_1']].mean().reset_index()
    resultados_age_1['Bondad age_1'] = resultados_age_1['Bondad age_1'].round(2)

    resultados_am1 = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad am1', 'Tiempo am1']].mean().reset_index()
    resultados_am1['Bondad am1'] = resultados_am1['Bondad am1'].round(2)

    resultados_am2 = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad am2', 'Tiempo am2']].mean().reset_index()
    resultados_am2['Bondad am2'] = resultados_am2['Bondad am2'].round(2)

    resultados_am3 = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad am3', 'Tiempo am3']].mean().reset_index()
    resultados_am3['Bondad am3'] = resultados_am3['Bondad am3'].round(2)

    if conf.MEJORA_BL:
        resultados_BL_mejora = tabla.groupby(['Tamaño', 'Densidad'])[['Bondad BL+', 'Tiempo BL+']].mean().reset_index()
        resultados_BL_mejora['Bondad BL+'] = resultados_BL_mejora['Bondad BL+'].round(2)


    print('')
    print('')
    print('[*][·][]Tabla Resultados para BL')
    print(resultados_BL)
    print('')
    print('[*][·][]Tabla Resultados para Greedy')
    print(resultados_Greedy)
    print('')
    print('[*][·][]Tabla Resultados para agg')
    print(resultados_agg)
    print('')
    print('[*][·][]Tabla Resultados para agg_1')
    print(resultados_agg_1)
    print('')
    print('[*][·][]Tabla Resultados para age')
    print(resultados_age)
    print('')
    print('[*][·][]Tabla Resultados para age_1')
    print(resultados_age_1)
    print('')
    print('[*][·][]Tabla Resultados para am1')
    print(resultados_am1)
    print('')
    print('[*][·][]Tabla Resultados para am2')
    print(resultados_am2)
    print('')
    print('[*][·][]Tabla Resultados para am3')
    print(resultados_am3)
    print('')
    if conf.MEJORA_BL:
        print('[*][·][]Tabla Resultados para BL+')
        print(resultados_BL_mejora)
        print('')

    for i in range(k,p):
        #-------------------
        # Filtrar los DataFrames por el tamaño de interés (en este caso, 100)
        resultados_BL_c = resultados_BL[resultados_BL['Tamaño'] == i*100]
        resultados_Greedy_c = resultados_Greedy[resultados_Greedy['Tamaño'] == i*100]
        resultados_agg_c = resultados_agg[resultados_agg['Tamaño'] == i*100]
        resultados_agg_1_c = resultados_agg_1[resultados_agg_1['Tamaño'] == i*100]
        resultados_age_c = resultados_age[resultados_age['Tamaño'] == i*100]
        resultados_age_1_c = resultados_age_1[resultados_age_1['Tamaño'] == i*100]
        resultados_am1_c = resultados_am1[resultados_am1['Tamaño'] == i*100]
        resultados_am2_c = resultados_am2[resultados_am2['Tamaño'] == i*100]
        resultados_am3_c = resultados_am3[resultados_am3['Tamaño'] == i*100]

        # Calcular las medias de Fitness  para cada algoritmo en tamaño 100
        media_BL_c = resultados_BL_c['Bondad BL'].mean().round(2)
        media_Greedy_c = resultados_Greedy_c['Bondad Greedy'].mean().round(2)
        media_agg_c = resultados_agg_c['Bondad agg'].mean().round(2)
        media_agg_1_c = resultados_agg_1_c['Bondad agg_1'].mean().round(2)
        media_age_c = resultados_age_c['Bondad age'].mean().round(2)
        media_age_1_c = resultados_age_1_c['Bondad age_1'].mean().round(2)
        media_am1_c = resultados_am1_c['Bondad am1'].mean().round(2)
        media_am2_c = resultados_am2_c['Bondad am2'].mean().round(2)
        media_am3_c = resultados_am3_c['Bondad am3'].mean().round(2)

        # La media de Tiempo no se redondea, ya que solo queremos redondear la Bondad
        media_tiempo_BL_c = resultados_BL_c['Tiempo BL'].mean()
        media_tiempo_Greedy_c = resultados_Greedy_c['Tiempo Greedy'].mean()
        media_tiempo_agg_c = resultados_agg_c['Tiempo agg'].mean()
        media_tiempo_agg_1_c = resultados_agg_1_c['Tiempo agg_1'].mean()
        media_tiempo_age_c = resultados_age_c['Tiempo age'].mean()
        media_tiempo_age_1_c = resultados_age_1_c['Tiempo age_1'].mean()
        media_tiempo_am1_c = resultados_am1_c['Tiempo am1'].mean()
        media_tiempo_am2_c = resultados_am2_c['Tiempo am2'].mean()
        media_tiempo_am3_c = resultados_am3_c['Tiempo am3'].mean()

        if conf.MEJORA_BL:
            resultados_BL_mejora_c = resultados_BL_mejora[resultados_BL_mejora['Tamaño'] == i*100]
            media_BL_mejora_c = resultados_BL_mejora_c['Bondad BL+'].mean().round(2)
            media_tiempo_BL_mejora_c = resultados_BL_mejora_c['Tiempo BL+'].mean()
            # Crear un nuevo DataFrame para mostrar los resultados
            resultados_globales_c = pd.DataFrame({
                'Algoritmo': ['BL', 'Greedy', 'Agg', 'Agg_1', 'Age', 'Age_1', 'Am1', 'Am2', 'Am3', 'BL+'],
                'Fitness': [media_BL_c, media_Greedy_c, media_agg_c, media_agg_1_c, media_age_c, media_age_1_c, media_am1_c, media_am2_c, media_am3_c, media_BL_mejora_c],
                'Tiempo': [media_tiempo_BL_c, media_tiempo_Greedy_c, media_tiempo_agg_c, media_tiempo_agg_1_c, media_tiempo_age_c, media_tiempo_age_1_c, media_tiempo_am1_c, media_tiempo_am2_c, media_tiempo_am3_c, media_tiempo_BL_mejora_c]
            })
        else:
            # Crear un nuevo DataFrame para mostrar los resultados
            resultados_globales_c = pd.DataFrame({
                'Algoritmo': ['BL', 'Greedy', 'Agg', 'Agg_1', 'Age', 'Age_1', 'Am1', 'Am2', 'Am3'],
                'Fitness': [media_BL_c, media_Greedy_c, media_agg_c, media_agg_1_c, media_age_c, media_age_1_c, media_am1_c, media_am2_c, media_am3_c],
                'Tiempo': [media_tiempo_BL_c, media_tiempo_Greedy_c, media_tiempo_agg_c, media_tiempo_agg_1_c, media_tiempo_age_c, media_tiempo_age_1_c, media_tiempo_am1_c, media_tiempo_am2_c, media_tiempo_am3_c]
            })

        # Si es necesario, ordena la tabla por la columna 'Fitness'
        resultados_globales_c = resultados_globales_c.sort_values(by='Fitness', ascending=False).reset_index(drop=True)

        print('')
        print('[*][·][]Tabla Resultados para Tamaño = {}'.format(100*i))
        # Mostrar la tabla de resultados globales para tamaño 100
        print(resultados_globales_c)

    
    
    if conf.HACER_TXT_DE_TABLA_EJECUCION:
        print('[*][·][]Tabla con todos los problemas')
        print(tabla)
        tabla.to_csv('ejecucion_completa.txt', sep='\t', index=False)


#   _____            __ _               
#  / ____|          / _(_)              
# | |  __ _ __ __ _| |_ _  ___ __ _ ___ 
# | | |_ | '__/ _` |  _| |/ __/ _` / __|
# | |__| | | | (_| | | | | (_| (_| \__ \
#  \_____|_|  \__,_|_| |_|\___\__,_|___/

    if conf.MOSTRAR_GRAFICAS:
        # Establecer el estilo de la gráfica
        plt.style.use('ggplot')

        # Tamaño del gráfico
        plt.figure(figsize=(14, 7))

        # Anchura de las barras
        bar_width = 0.10

        # Índices de las barras
        indices = np.arange(len(tabla['Nombre del problema']))

        # Dibujo de las barras
        plt.bar(indices, tabla['Bondad BL'], bar_width, label='Bondad BL', alpha=0.8)
        plt.bar(indices + bar_width, tabla['Bondad Greedy'], bar_width, label='Bondad Greedy', alpha=0.8)
        plt.bar(indices + bar_width*2, tabla['Bondad agg'], bar_width, label='Bondad agg', alpha=0.8)
        plt.bar(indices + bar_width*3, tabla['Bondad agg_1'], bar_width, label='Bondad agg_1', alpha=0.8)
        plt.bar(indices + bar_width*4, tabla['Bondad age'], bar_width, label='Bondad age', alpha=0.8)
        plt.bar(indices + bar_width*5, tabla['Bondad age_1'], bar_width, label='Bondad age_1', alpha=0.8)
        plt.bar(indices + bar_width*6, tabla['Bondad am1'], bar_width, label='Bondad am1', alpha=0.8)
        plt.bar(indices + bar_width*7, tabla['Bondad am2'], bar_width, label='Bondad am2', alpha=0.8)
        plt.bar(indices + bar_width*8, tabla['Bondad am3'], bar_width, label='Bondad am3', alpha=0.8)
        if conf.MEJORA_BL:
            plt.bar(indices + bar_width * 9, tabla['Bondad BL+'], bar_width, label='Bondad BL+', alpha=0.8)

        # Añadir títulos y etiquetas
        plt.xlabel('Nombre del problema')
        plt.ylabel('Bondad')
        plt.title('Comparación de Bondad')
        plt.xticks(indices + bar_width / 2, tabla['Nombre del problema'], rotation=90)

        # Añadir leyenda
        plt.legend()

        # Mostrar gráfico
        plt.tight_layout()
        plt.show()
        
        #-----------------------------
        
        # Establecer el estilo de la gráfica
        plt.style.use('ggplot')

        # Tamaño del gráfico
        plt.figure(figsize=(14, 7))

        # Anchura de las barras
        bar_width = 0.10

        # Índices de las barras
        indices = np.arange(len(tabla['Nombre del problema']))

        # Dibujo de las barras
        plt.bar(indices, tabla['Tiempo BL'], bar_width, label='Tiempo BL', alpha=0.8)
        plt.bar(indices + bar_width, tabla['Tiempo Greedy'], bar_width, label='Tiempo Greedy', alpha=0.8)
        plt.bar(indices + bar_width*2, tabla['Tiempo agg'], bar_width, label='Tiempo agg', alpha=0.8)
        plt.bar(indices + bar_width*3, tabla['Tiempo agg_1'], bar_width, label='Tiempo agg_1', alpha=0.8)
        plt.bar(indices + bar_width*4, tabla['Tiempo age'], bar_width, label='Tiempo age', alpha=0.8)
        plt.bar(indices + bar_width*5, tabla['Tiempo age_1'], bar_width, label='Tiempo age_1', alpha=0.8)
        plt.bar(indices + bar_width*6, tabla['Tiempo am1'], bar_width, label='Tiempo am1', alpha=0.8)
        plt.bar(indices + bar_width*7, tabla['Tiempo am2'], bar_width, label='Tiempo am2', alpha=0.8)
        plt.bar(indices + bar_width*8, tabla['Tiempo am3'], bar_width, label='Tiempo am3', alpha=0.8)
        if conf.MEJORA_BL:
            plt.bar(indices + bar_width*9, tabla['Tiempo BL+'], bar_width, label='Tiempo BL+', alpha=0.8)

        # Añadir títulos y etiquetas
        plt.xlabel('Nombre del problema')
        plt.ylabel('Tiempo')
        plt.title('Comparación de Tiempo')
        plt.xticks(indices + bar_width / 2, tabla['Nombre del problema'], rotation=90)

        # Añadir leyenda
        plt.legend()

        # Mostrar gráfico
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    main()