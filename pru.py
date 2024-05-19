import algoritmosV3 as alg
import numpy as np
import config as conf
import os

ruta_archivo = "data/jeu_{}_{}_{}.txt".format(200, 50, 4)  # Reemplaza con la ruta de tu archivo

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
sol_am1, eva_am1, bl_am1 = alg.agg(matriz_valores, peso_maximo, vector_pesos, meme=1,cruce=1)
print("Am1_agg_1-> ({}) Con un peso disponible de: {}    y bondad total de: {} || evaluaciones P: {};  evaluaciones bl: {}".format(ruta_archivo, peso_maximo - sol_am1.peso ,sol_am1.beneficio, eva_am1-bl_am1, bl_am1))
print('')
np.random.seed(conf.SEMILLA)
sol_am2, eva_am2, bl_am2 = alg.agg(matriz_valores, peso_maximo, vector_pesos, meme=2,cruce=1)
print("Am2_agg_1 -> ({}) Con un peso disponible de: {}    y bondad total de: {} || evaluaciones P: {};  evaluaciones bl: {}".format(ruta_archivo, peso_maximo - sol_am2.peso ,sol_am2.beneficio, eva_am2-bl_am2, bl_am2))
print('')
np.random.seed(conf.SEMILLA)
sol_am3, eva_am3, bl_am3 = alg.agg(matriz_valores, peso_maximo, vector_pesos, meme=3,cruce=1)
print("Am3_agg_1 -> ({}) Con un peso disponible de: {}    y bondad total de: {} || evaluaciones P: {};  evaluaciones bl: {}".format(ruta_archivo, peso_maximo - sol_am3.peso ,sol_am3.beneficio, eva_am3-bl_am3, bl_am3))
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
sol_am1, eva_am1, bl_am1 = alg.age_AM(matriz_valores, peso_maximo, vector_pesos, meme=1,cruce=1)
print("Am1_age_1-> ({}) Con un peso disponible de: {}    y bondad total de: {} || evaluaciones P: {};  evaluaciones bl: {}".format(ruta_archivo, peso_maximo - sol_am1.peso ,sol_am1.beneficio, eva_am1-bl_am1, bl_am1))
print('')
np.random.seed(conf.SEMILLA)
sol_am2, eva_am2, bl_am2 = alg.age_AM(matriz_valores, peso_maximo, vector_pesos, meme=2,cruce=1)
print("Am2_age_1 -> ({}) Con un peso disponible de: {}    y bondad total de: {} || evaluaciones P: {};  evaluaciones bl: {}".format(ruta_archivo, peso_maximo - sol_am2.peso ,sol_am2.beneficio, eva_am2-bl_am2, bl_am2))
print('')
np.random.seed(conf.SEMILLA)
sol_am3, eva_am3, bl_am3 = alg.age_AM(matriz_valores, peso_maximo, vector_pesos, meme=3,cruce=1)
print("Am3_age_1 -> ({}) Con un peso disponible de: {}    y bondad total de: {} || evaluaciones P: {};  evaluaciones bl: {}".format(ruta_archivo, peso_maximo - sol_am3.peso ,sol_am3.beneficio, eva_am3-bl_am3, bl_am3))
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