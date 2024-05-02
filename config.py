#   _____ ____  _   _ ______ _____ _____ _    _ _____            _____ _____ ____  _   _ 
#  / ____/ __ \| \ | |  ____|_   _/ ____| |  | |  __ \     /\   / ____|_   _/ __ \| \ | |
# | |   | |  | |  \| | |__    | || |  __| |  | | |__) |   /  \ | |      | || |  | |  \| |
# | |   | |  | | . ` |  __|   | || | |_ | |  | |  _  /   / /\ \| |      | || |  | | . ` |
# | |___| |__| | |\  | |     _| || |__| | |__| | | \ \  / ____ \ |____ _| || |__| | |\  |
#  \_____\____/|_| \_|_|    |_____\_____|\____/|_|  \_\/_/    \_\_____|_____\____/|_| \_|

#semilla aleatoria
SEMILLA = 777
MAX_EVALUACIONES = 90000
ENTENDER_FACTIVILIDAD_DE_SOLUCION_COMO_PARTE_FUNCION_OBJETIVO = False
# ^ Si esta a true la ajecucion de todos los problemas con MEJORA_BL tarda unos 5 min | si False tarda 15 min
MEJORA_BL = True

MOSTRAR_CADA_SALIDA = True
MOSTRAR_GRAFICAS = True

SOLO_EJECUTAR_100 = True
SOLO_EJECUTAR_200 = False
SOLO_EJECUTAR_300 = False

HACER_TXT_DE_TABLA_EJECUCION = False

VER_GRAFICA_DE_MEJORA_SOLO_PARA_UN_PROBLEMA = False 
PROBLEMA_PARA_VER_GRAFICA = 'data/jeu_300_50_5.txt'

CLEAR = 'clear' #comando para limpiar pantalla en tu SO


POBLACION = 50

PROBABILIDAD_CRUZE = 0.68
PROBABILIDAD_MUTACION = 0.08
PROBABILIDAD_MEME2 = 0.1
