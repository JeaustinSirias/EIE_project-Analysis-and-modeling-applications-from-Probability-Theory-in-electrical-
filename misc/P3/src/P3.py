import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fitter import Fitter
from scipy import stats
import tikzplotlib as tikz


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

def distribucion_conjunta(X, Y, bins):
    '''
    Pide por parámetros dos variables aleatorias
    X y Y, así como el número de 'bins' o divisiones
    a emplear para construir el histograma bivariado.
    Retorna
    '''
    np.seterr(all='ignore') # ignorar advertencias
    
    # Se inicializa la figura interactiva 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Se obtiene el plano de probabilidades para graficar el hist3D
    hist, xbins, ybins = np.histogram2d(X, Y, bins=bins, normed=False)
    hist = hist / sum(sum(hist))
    xbins = (xbins + np.roll(xbins, -1))[:-1] / 2.0 
    ybins = (ybins + np.roll(ybins, -1))[:-1] / 2.0 
    
    #Formatos de retorno para la funcion de densidad bivariada discreta
    xyp = [[xbins[i], ybins[j], hist[i][j]] for i in range(bins) for j in range(bins)]
    xy = hist 

    # Se construyen los arreglos para el ancho de Bins * Bins barras
    xpos, ypos = np.meshgrid(xbins, ybins, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Se dimensiona el ancho visual de las barras (como un sólido).
    dx = dy = 30 * np.ones_like(zpos)
    dz = hist.ravel() 

    # Se visualiza el histograma 3D
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    ax.set_xlabel('La hora 1 (X)')
    ax.set_ylabel('La hora 2 (Y)')
    ax.set_zlabel('Probabilidad')
    plt.show()

    return xyp, xy, xbins, ybins

def evaluar_modelos(datos):
    '''
    Evalúa las 80 distribuciones del módulo stats
    y obtiene el el modelo de mejor ajuste para
    la demanda anual de una hora específica. Retorna
    el nombre de la mejor distribución y una tupla con
    los parámetros del modelo.
    '''
    np.seterr(all='ignore') # ignorar advertencias
    
    # Hallar el mejor ajuste con Fitter
    f = Fitter(datos, timeout=120)
    f.fit()
    ajuste = f.get_best()
    
    for i in ajuste.keys():
        dist, params = i, ajuste[i]

    print('------------\nDistribución\n------------')
    print(dist, '\n')
    print('----------\nParámetros\n----------')
    print(params)

    return dist, params


def densidad_marginal(xy, bins, dist, params, eje):
    '''
    Se elije eje='x' o eje='y' segun sea el caso para la 
    densidad marginal en Y o en X. El parámetro 'xy' es el
    formato de datos bivariable, 'bins' es el vector de valores
    de potencia xbins o ybins. Los parámetros 'dist' y 'params'
    corresponden al modelo de mejor ajuste retornado por el fitter.
    '''
    np.seterr(all='ignore') # ignorar advertencias
    
    # Hallar la densidad marginal de x o y, según se indique en 'eje'
    if eje == 'x':

        filas = len(xy)
        marginal = [sum(xy[i]) for i in range(filas)]

    elif eje == 'y':

        xy = xy.transpose()
        filas = len(xy)
        marginal = [sum(xy[i]) for i in range(filas)]

    # Visualizar modelo de mejor ajuste
    distro = getattr(stats, dist) 
    d = np.arange(min(bins)*0.96, max(bins)*1.04, 1)
    pdf_plot = distro.pdf(d, *params)
    plt.plot(d, pdf_plot*22, lw=3.5, color='r')
    
    # Visualizar función de densidad marginal
    plt.bar(bins, marginal, width=12)
    plt.title('Contraste: densidad marginal vs. modelo de mejor ajuste')
    plt.xlabel('Potencia [MW]')
    plt.ylabel('Densidad Probabilística')
	#tikz.save('marginal.tikz'))
    return marginal


# Se eligen las dos horas que desean estudiarse
hora_1 = extraer_datos('demanda_2019.json', 3) 
hora_2 = extraer_datos('demanda_2019.json', 11) 

# Se ejecuta el análisis bivariado
xyp, xy, xbins, ybins = distribucion_conjunta(hora_1, hora_2, bins = 10)

# Se llama obtiene los parámetros de mejor ajuste para cada hora

#dist, params = evaluar_modelos(hora_1)
#_dist, _params = evaluar_modelos(hora_2)

# Se contrasta la densidad marginal para cada hora, según su eje
#densidad_marginal(xy, xbins, dist, params, eje='x')