#!/usr/bin/python3

#importando los paquetes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as spsts
import datetime
from scipy import stats


#=================================FUNCIONES==================================
def acondicionar_datos(json, tiempo):
	df =  pd.read_json(json) #se carga el frame de datos
	df = np.array(df) #se convierte el frame en un array
	
	#se extraen las fechas y las demandas en arreglos separados
	muestras = len(df)
	vector_fecha = []; demanda = []
	for datos in range(muestras):
		
		hora = df[datos][0]['fechaHora']
		#if int(hora[11:13]) == tiempo:
		demanda.append(df[datos][0]['MW'])
	
	return demanda
		
#===============================================================================
def evaluar_modelos(datos, dists, Bins, hora):
	#condiciones iniciales
	rmse_min = np.inf
	p_max = 0 #El mejor p en chisqr test
	kspmax = 0 #El mejor p en KStest
	np.seterr(all = 'ignore')

	#Se prepara el espacio de visualizacion:
	fig, ax = plt.subplots(1, 3, figsize = (16, 5), tight_layout = True)
	#fig1:
	ax[0].set_title('Distribución observada: demanda a las {}'.format(datetime.time(hora)))
	ax[0].set_xlabel('Potencia [MW]')
	ax[0].set_ylabel('Frecuencia')
	#fig2:
	ax[1].set_title('Ajuste por funciones de densidad')
	ax[1].set_ylabel('Frecuencia')
	ax[1].set_xlabel('Potencia [MW]')
	#fig3:
	ax[2].set_title('Mejor ajuste basado en criterios de bondad')
	ax[2].set_ylabel('Frecuencia')
	ax[2].set_xlabel('Potencia [MW]')
	
	#Distribucion observada:
	ocurrencias_exp, bins = np.histogram(datos, bins = Bins)
	for i in range(Bins):
		if ocurrencias_exp[i] == 0:
			ocurrencias_exp[i] = 1

	bins_centrados = (bins + np.roll(bins, -1))[:-1] / 2.0 
	escala = len(datos) * (max(datos) - min(datos)) / len(bins_centrados)
	
	#Probando las distribuciones ingresadas
	for distribucion in dists:
		dist = getattr(spsts, distribucion)
		param = dist.fit(datos)
		pdf = dist.pdf(bins_centrados, *param)
		pdf_plot = dist.pdf(np.arange(min(datos) * 0.96, max(datos) * 1.04, 1), *param)
		ocurrencias_teo = [int(round(freq)) for freq in escala * pdf]
		ax[1].plot(np.arange(min(datos) * 0.96, max(datos) * 1.04, 1), escala * pdf_plot, lw = 3.5, label = '{}'.format(distribucion))
		
		#Bondad de ajuste por chisquare:
		coef_chi, p = spsts.chisquare(f_obs = ocurrencias_teo, f_exp = ocurrencias_exp)
		if p > p_max:
			p_max = p
			dist_chi = distribucion
			mod_chi = dist, param, pdf
		
		#Bondad de ajuste por RMSE(Root-Mean-Square Error):
		diferencia = (ocurrencias_teo - ocurrencias_exp)**2
		rmse = np.sqrt(np.mean(diferencia))
		if rmse < rmse_min:
			rmse_min = rmse
			dist_rmse = distribucion
			mod_rmse = dist, param, pdf
	
		#Bondad de ajuste por Kolvogorov-Smirnov:
		D, ksp = spsts.kstest(datos, distribucion, args = param)
		if ksp > kspmax:
			kspmax = ksp
			dist_ks = distribucion

	#visualizando resultados:
	ax[0].hist(datos, bins = Bins, color = 'tomato', histtype='bar', rwidth=0.8)
	ax[1].hist(datos, bins = Bins, color = 'palevioletred', histtype='bar', rwidth=0.8)
	ax[2].hist(datos, bins = Bins, color = 'b')
		
	if dist_chi == dist_rmse or dist_chi == dist_ks:
		params = mod_chi[1]
		mejor_ajuste = dist_chi
		ax[2].hist(datos, bins = Bins, color = 'cornflowerblue', label = 'Distribución observada')
		ax[2].bar(bins_centrados, mod_chi[2] * escala, width = 6, color = 'r', label = 'Mejor ajuste: {}'.format(dist_chi))
		m, v, s, k = mod_chi[0].stats(*params, moments = 'mvsk') 
		
	elif dist_rmse == dist_ks:
		params = mod_rmse[1]
		mejor_ajuste = dist_rmse
		ax[2].hist(datos, bins = Bins, color = 'cornflowerblue', label = 'Distribución observada')	
		ax[2].bar(bins_centrados, mod_rmse[2] * escala, width = 6, color = 'r', label = 'Mejor ajuste: {}'.format(dist_rmse))
		m, v, s, k = mod_rmse[0].stats(*params, moments = 'mvsk')
		
		
		
	#imprimiendo resumen y resultados:

	print('Resumen:\nEl mejor ajuste por RMSE ocurre con la distribución', dist_rmse)
	print('El mejor ajuste por chisquare ocurre con la distribución', dist_chi)
	print('El mejor ajuste por Kolmogorov ocurre con la distribución', dist_ks)
	print('Mejor modelo ajustada por criterios de bondad:', mejor_ajuste)
	print('Cantidad de muestras:', len(datos))	
	print('.\n.\n.\n.\nMomentos centrales para el mejor ajuste:', '\nMedia:', m, '\nVarianza:', v, '\nCoef. Simetría:', s, '\nCurtosis:', k)
	ax[1].legend()
	ax[2].legend()
	plt.show()
	
#================================================================================================

'''
distribuciones = ['norm', 'rayleigh', 'expon', 'uniform', 'burr12', 'alpha', 'gamma', 'beta', 'pareto']
hora = 17
demandas = acondicionar_datos('./database/data.json', hora)
demandas = np.array(demandas)
x = range(len(demandas))
plt.figure(tight_layout = True)
plt.plot(x, demandas, color = 'tab:green')
plt.title('Demanda energética nacional por hora desde el 01-01-2019 al 13-09-2019', fontsize =  16)
plt.ylabel('Demanda [KW]', fontsize = 14)
plt.xlabel('Tiempo [h]', fontsize =  14)
plt.grid()
plt.show()
#modelo = evaluar_modelos(demandas, distribuciones, 25, hora)
'''







