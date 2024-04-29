import algoritmosV3 as alg
import os

ruta_archivo = "data/jeu_{}_{}_{}.txt".format(300, 25, 1)  # Reemplaza con la ruta de tu archivo

matriz_valores, peso_maximo, vector_pesos = alg.procesar_archivo(ruta_archivo)

print("greedi: {}".format(alg.greedy(matriz_valores, peso_maximo, vector_pesos).beneficio))
print("agg: {}".format(alg.agg(matriz_valores, peso_maximo, vector_pesos).beneficio))