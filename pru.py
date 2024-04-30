import algoritmosV3 as alg
import os

ruta_archivo = "data/jeu_{}_{}_{}.txt".format(300, 25, 1)  # Reemplaza con la ruta de tu archivo

matriz_valores, peso_maximo, vector_pesos = alg.procesar_archivo(ruta_archivo)

print("greedi: {}".format(alg.greedy(matriz_valores, peso_maximo, vector_pesos).beneficio))
print("agg: {}".format(alg.agg(matriz_valores, peso_maximo, vector_pesos).beneficio))
print("agg_1: {}".format(alg.agg(matriz_valores, peso_maximo, vector_pesos,cruce=1).beneficio))
print("Am1: {}".format(alg.agg(matriz_valores, peso_maximo, vector_pesos,meme=1).beneficio))
print("Am2: {}".format(alg.agg(matriz_valores, peso_maximo, vector_pesos,meme=2).beneficio))
print("Am3: {}".format(alg.agg(matriz_valores, peso_maximo, vector_pesos,meme=3).beneficio))
print("age: {}".format(alg.age(matriz_valores, peso_maximo, vector_pesos).beneficio))
print("age_1: {}".format(alg.age(matriz_valores, peso_maximo, vector_pesos,cruce=1).beneficio))