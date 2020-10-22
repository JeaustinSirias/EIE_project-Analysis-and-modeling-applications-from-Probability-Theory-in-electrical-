#!/usr/bin/python3
import numpy as np
from fitter import Fitter
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime



#====================================================================
def extraer_datos(archivo_json, hora):
    '''Importa la base de datos completa y devuelve los
    datos de potencia a la hora indicada en un
    array de valores.
    '''
    
    # Cargar el "DataFrame"
    df = pd.read_json(archivo_json) 
    
    # Convertir en un array de NumPy
    datos = np.array(df)                

    # Crear vector con los valores demanda en una hora
    demanda = []

    # Extraer la demanda en la hora seleccionada
    for i in range(len(datos)):
        instante = datetime.fromisoformat(datos[i][0]['fechaHora'])
        if instante.hour == hora:
            demanda.append(datos[i][0]['MW'])

    return demanda
#====================================================================
def mejor_ajuste(datos):
	f = Fitter(datos)
	f.fit()
	ajuste = f.get_best()

	for i in ajuste.keys():
		dist, params = i, ajuste[i]

	print('------------------\nMejor distribución\n------------------')
	print(dist, '\n')
	print('----------\nParámetros\n----------')
	print(params)

	return dist, params

#====================================================================
def distribucion_conjunta(h1, h2, Bins):

	# Se inicializa la figura interactiva 3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# Se obtiene el plano de probabilidades para graficar el hist3D
	hist, xbins, ybins = np.histogram2d(h1, h2, bins=Bins, normed=False)
	xbins = (xbins + np.roll(xbins, -1))[:-1] / 2.0 
	ybins = (ybins + np.roll(ybins, -1))[:-1] / 2.0 
	escala = len(h1) * (max(h1) - min(h1)) / len(xbins)


	#Formatos de retorno para la funcion de densidad bivariada discreta
	xyp = [[xbins[i], ybins[j], hist[i][j]] for i in range(Bins) for j in range(Bins)]
	xy = hist

	# Se construyen los arreglos para el ancho de 10 * 10 barras
	xpos, ypos = np.meshgrid(xbins, ybins, indexing="ij")
	xpos = xpos.ravel()
	ypos = ypos.ravel()
	zpos = 0

	# Se dimensiona el ancho visual de las barras (como un sólido).
	dx = dy = 30 * np.ones_like(zpos)
	dz = hist.ravel() 

	# Se visualiza el histograma 3D
	ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
	ax.set_xlabel('Hora1')
	ax.set_ylabel('Hora2')
	ax.set_zlabel('Probabilidad')
	plt.show()

	return xyp, xy, xbins, ybins, escala
#====================================================================
def densidad_marginal(xy, bins, dist, params, escala, eje):

	# Hallar la densidad marginal de x o y, según se indique en 'eje'
	if eje == 'x':

		filas = len(xy)
		marginal = [sum(xy[i]) for i in range(filas)]

	elif eje == 'y':

		xy = xy.transpose()
		filas = len(xy)
		marginal = [sum(xy[i]) for i in range(filas)]


	distro = getattr(stats, dist) 
	d = np.arange(min(bins)*0.96, max(bins)*1.04, 1)
	pdf_plot = distro.pdf(d, *params)
	plt.plot(d, pdf_plot*escala, lw=3.5)
	plt.bar(bins, marginal, width = 12)

	plt.show()
	return marginal

#====================================================================\
def energia_diaria(archivo_json):

	# Cargar el "DataFrame"
	df = pd.read_json(archivo_json) 

	# Convertir en un array de NumPy
	datos = np.array(df)  

	# Crear vector con todos los valores horarios de demanda
	demanda = []

	# Extraer la magnitud de la demanda para todas las horas
	for hora in range(len(datos)):
		instante = datetime.fromisoformat(datos[hora][0]['fechaHora'])
		demanda.append(datos[hora][0]['MW'])

	# Separar las magnitudes en grupos de 24 (24 h)
	demanda = np.split(np.array(demanda), len(demanda) / 24)

	#Crear vector para almacenar la enegia a partir de la demanda
	energia = []

	#calcular la energia diaria por la Regla del Trapecio
	for dia in range(len(demanda)):

		E = round(np.trapz(demanda[dia]), 2)
		energia.append(E)

	return energia
#====================================================================
def parametros_energia(vector_energia):

	media = np.median(vector_energia)
	desviacion = np.std(vector_energia)

	return media, desviacion
#====================================================================

data1 = extraer_datos('demanda_2019.json', 18) 
data2 = extraer_datos('demanda_2019.json', 11) 
#xyp, xy, xbins, ybins, escala = distribucion_conjunta(data1, data2, Bins=10)


#dist, params = mejor_ajuste(data1)
#densidad_marginal(xy, xbins, dist, params, escala, eje='x')


d = {'Y1':[0.03, 0.09, 0.005, 0.06], 'Y2':[0.04, 0.0056, 0.045, 0.023], 'Y3':[0.04, 0.04, 0.069, 0.0025], 'Y4':[0.04, 0.086, 0.001, 0.0014], 'Y5':[0.04, 0.054, 0.001, 0.084], 'Y6':[0.04, 0.004, 0.06, 0.05] }
df = pd.DataFrame(data=d)

print(df)


