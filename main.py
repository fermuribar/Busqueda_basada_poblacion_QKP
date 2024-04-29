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
        np.random.seed(conf.SEMILLA)
        solucion_mejora = alg.BL_primer_mejor(matriz_valores,peso_maximo,vector_pesos, vecindario = 1)
        return

    # Crear un DataFrame vacío con las columnas específicas
    if conf.MEJORA_BL:
        columnas = ['Nombre del problema', 'Solucion BL', 'Peso BL', 'Bondad BL', 'Tiempo BL', 'Solucion Greedy', 'Peso Greedy', 'Bondad Greedy', 'Tiempo Greedy', 'Peso agg', 'Bondad agg', 'Tiempo agg', 'Solucion B+', 'Peso BL+', 'Bondad BL+', 'Tiempo BL+']
    else:
        columnas = ['Nombre del problema', 'Solucion BL', 'Peso BL', 'Bondad BL', 'Tiempo BL', 'Solucion Greedy', 'Peso Greedy', 'Bondad Greedy', 'Tiempo Greedy', 'Peso agg', 'Bondad agg', 'Tiempo agg']
        
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

                inicio = time.time()
                sol_agg = alg.agg(matriz_valores, peso_maximo, vector_pesos)
                fin = time.time()
                duracion_agg = fin - inicio
                if conf.MOSTRAR_CADA_SALIDA:
                    print(sol.solucion.astype(int))
                    print("Agg -> ({}) Con un peso disponible de: {}    y bondad total de: {} ||T {}".format(ruta_archivo, peso_maximo - sol_agg.peso ,sol_agg.beneficio, duracion_agg))
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
                        
                    tabla.loc[len(tabla)] = [ruta_archivo, solucion.solucion, solucion.peso, solucion.beneficio, duracion_BL, sol.solucion, sol.peso, sol.beneficio, duracion_Greedy, sol_agg.peso, sol_agg.beneficio, duracion_agg, solucion_mejora.solucion, solucion_mejora.peso, solucion_mejora.beneficio, duracion_BL_mejora]
                else:
                    tabla.loc[len(tabla)] = [ruta_archivo, solucion.solucion, solucion.peso, solucion.beneficio, duracion_BL, sol.solucion, sol.peso, sol.beneficio, duracion_Greedy, sol_agg.peso, sol_agg.beneficio, duracion_agg]
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

        # Calcular las medias de Fitness  para cada algoritmo en tamaño 100
        media_BL_c = resultados_BL_c['Bondad BL'].mean().round(2)
        media_Greedy_c = resultados_Greedy_c['Bondad Greedy'].mean().round(2)
        media_agg_c = resultados_agg_c['Bondad agg'].mean().round(2)

        # La media de Tiempo no se redondea, ya que solo queremos redondear la Bondad
        media_tiempo_BL_c = resultados_BL_c['Tiempo BL'].mean()
        media_tiempo_Greedy_c = resultados_Greedy_c['Tiempo Greedy'].mean()
        media_tiempo_agg_c = resultados_agg_c['Tiempo agg'].mean()

        if conf.MEJORA_BL:
            resultados_BL_mejora_c = resultados_BL_mejora[resultados_BL_mejora['Tamaño'] == i*100]
            media_BL_mejora_c = resultados_BL_mejora_c['Bondad BL+'].mean().round(2)
            media_tiempo_BL_mejora_c = resultados_BL_mejora_c['Tiempo BL+'].mean()
            # Crear un nuevo DataFrame para mostrar los resultados
            resultados_globales_c = pd.DataFrame({
                'Algoritmo': ['BL', 'Greedy', 'Agg', 'BL+'],
                'Fitness': [media_BL_c, media_Greedy_c, media_agg_c, media_BL_mejora_c],
                'Tiempo': [media_tiempo_BL_c, media_tiempo_Greedy_c, media_tiempo_agg_c, media_tiempo_BL_mejora_c]
            })
        else:
            # Crear un nuevo DataFrame para mostrar los resultados
            resultados_globales_c = pd.DataFrame({
                'Algoritmo': ['BL', 'Greedy', 'Agg'],
                'Fitness': [media_BL_c, media_Greedy_c, media_agg_c],
                'Tiempo': [media_tiempo_BL_c, media_tiempo_Greedy_c, media_tiempo_agg_c]
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
        bar_width = 0.15

        # Índices de las barras
        indices = np.arange(len(tabla['Nombre del problema']))

        # Dibujo de las barras
        barras_bl = plt.bar(indices, tabla['Bondad BL'], bar_width, label='Bondad BL', alpha=0.8)
        barras_greedy = plt.bar(indices + bar_width, tabla['Bondad Greedy'], bar_width, label='Bondad Greedy', alpha=0.8)
        barras_agg = plt.bar(indices + bar_width*2, tabla['Bondad agg'], bar_width, label='Bondad agg', alpha=0.8)
        if conf.MEJORA_BL:
            barras_bl_mejora = plt.bar(indices + bar_width * 3, tabla['Bondad BL+'], bar_width, label='Bondad BL+', alpha=0.8)

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
        bar_width = 0.15

        # Índices de las barras
        indices = np.arange(len(tabla['Nombre del problema']))

        # Dibujo de las barras
        barras_bl = plt.bar(indices, tabla['Tiempo BL'], bar_width, label='Tiempo BL', alpha=0.8)
        barras_greedy = plt.bar(indices + bar_width, tabla['Tiempo Greedy'], bar_width, label='Tiempo Greedy', alpha=0.8)
        barras_agg = plt.bar(indices + bar_width*2, tabla['Tiempo agg'], bar_width, label='Tiempo agg', alpha=0.8)
        if conf.MEJORA_BL:
            barras_bl_mejora = plt.bar(indices + bar_width*3, tabla['Tiempo BL+'], bar_width, label='Tiempo BL+', alpha=0.8)

        # Añadir títulos y etiquetas
        plt.xlabel('Nombre del problema')
        plt.ylabel('Tiempo')
        plt.title('Comparación de Tiempo entre BL y Greedy')
        plt.xticks(indices + bar_width / 2, tabla['Nombre del problema'], rotation=90)

        # Añadir leyenda
        plt.legend()

        # Mostrar gráfico
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    main()