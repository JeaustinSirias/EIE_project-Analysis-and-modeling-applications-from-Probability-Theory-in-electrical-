#!/usr/bin/python3
#importing packages
import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#========Functions==========

######################################################################################################

def readData(inputDs, indexName): #indexName is a *kwarg

	data = pd.read_csv(inputDs, header = None, skiprows = 1, index_col = 0 )
	return np.array(data.loc['{}'.format(indexName)])

######################################################################################################

def computeFstDigit(vector):

	fstDigit = []
	strVect = []
	vecSize = np.arange(0, len(vector), 1)
	strSize = []

	#first we turn each element in vector into str
	for i in vecSize:

		num = vector[i]
		if num == 0: continue;
		else: strVect.append(str(num));

	strSize = np.arange(0, len(strVect), 1)

	#save the leading digit for each element. It cant be zero
	for j in strSize:

		if strVect[j][0] == '0': fstDigit.append(int(strVect[j][2]));
		else: fstDigit.append(int(strVect[j][0]));

	#finding out how many times 1,2, ..., 9 are repeated
	counters = [] #stores the times each leading digit gets repeated
	for k in np.arange(1, 10, 1):
		count = fstDigit.count(k)
		counters.append(count)
	totalSum = sum(counters)
	dataPerc = [i/totalSum for i in counters]

	return counters, dataPerc, totalSum
	#reading first digit for each element

######################################################################################################

def lookForExpectedCounts(countsNum):

	bfVals = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
	predictedVals = [round((k*countsNum)/100) for k in bfVals]

	return predictedVals, bfVals

######################################################################################################

def printSummary(dataset, ObsCounts, ExpCounts, pReal, bfVals):

	print("*****Summary Statistics******\n. \n. \n. \n. \n.")
	print('Observed counts: {}'.format(ObsCounts))
	print('Expected counts: {}\n. \n. \n. \n. \n.'.format(ExpCounts))
	print("Probabilities\n. \n. \n. \n. \n.")

	for i in np.arange(0, 9, 1):

		print('{num} Observed Prob: {p1}, Expected Prob: {p2}'.format(num = i +1, p1 = round(pReal[i], 3), p2 = round(bfVals[i]/100, 3)))

	#plotting
	plt.rcParams['axes.grid'] = True
	fig = plt.figure(tight_layout = True, figsize = (11, 8))
	figGrid = gridspec.GridSpec(2, 2)
	rawData = fig.add_subplot(figGrid[0, 0])
	dataHist = fig.add_subplot(figGrid[0, 1])
	benfordFreq = fig.add_subplot(figGrid[1, 0])
	benfordDist = fig.add_subplot(figGrid[1, 1])

	xData = np.arange(0, len(dataset), 1)
	xDigits = np.arange(1, 10, 1)

	#RawData plot
	rawData.plot(xData, dataset, color = 'tab:red')
	rawData.set_xlabel('Tiempo')
	rawData.set_ylabel('Variable física')
	rawData.set_title('Comportamiento histórico de la variable')
	

	#dataHist
	dataHist.hist(dataset, bins = 25, color = 'g', histtype='bar', rwidth=0.8)
	dataHist.set_ylabel('Frequencia')
	dataHist.set_xlabel('Variable física ')
	dataHist.set_title('Histograma')

	#benfordDist

	benfordDist.bar(xDigits, ObsCounts, color = 'teal', label = 'Rainfall: Observed occurrences per digit')

	for i in  xDigits:

		if i == xDigits[-1]:
			benfordDist.plot(i, ExpCounts[i-1], marker = 's', markersize = 8, color = 'firebrick', label = 'Benford\'s Law expected occurrences')
		else:
			benfordDist.plot(i, ExpCounts[i-1], marker = 's', markersize = 8, color = 'firebrick')

	benfordDist.set_xlabel('Leading digit')
	benfordDist.set_xticks(xDigits)
	benfordDist.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9'))
	benfordDist.set_ylabel('Number of occurrences')
	benfordDist.set_title('Leading digits distribution: Observed vs. Expected')
	benfordDist.legend()

	#benfordFreq plot
	benfordFreq.plot(xDigits, pReal*100, color = 'orangered', lw = 3, label = 'Rainfall: Observed probability')
	benfordFreq.plot(xDigits, bfVals, '--', color = 'tab:blue', lw = 3, label = 'Benford\'s  Law pedriction')
	benfordFreq.set_xlabel('Leading digit')
	benfordFreq.set_ylabel('Frequency [%]')
	benfordFreq.set_title('Leading digit probability')

	benfordFreq.legend()

	fig.align_labels()
	plt.show()

######################################################################################################


'''According to Benfords law, probability of occurrence for leading digit is
P_1stD = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]. cero is considered as null
Aditionally we can generalize more digits beyond the first one. For the second digit we have
the next probability.

P_2ndD = [12, 11.4, 10.9, 10.4, 10, 9.7, 9.3, 9, 8.8, 8.5] it doesnt consider zero as null.
'''

#SOLUCION AL LA PARTE 1:
'''Estudiar las probabilidades condicionales de ocurrencia del 2do digito dada la
ocurrencia del 1er digito

Aqui se requiere utilizar el concepto de probabilidad condicional P(A|B) = P(A n B) / P(B)
En este caso P(A|B) de entenderse como la probabilidad conjunta de que ocurran dos ciertos digitos
seguidos. La generalizacion de la ley de Benford puede extenderse para n en 10 <= n <= 99 con
P(n) = log10(n+1) - log10(n). Entonces se busca la probabilidad conjunta con la siguiente funcion:
'''
def probConjunta():
	
	digits = np.arange(10, 100, 1) #{10, 11, 12, ..., 99}
	probabilities = [np.log10(i+1) - np.log(i) for i in digits]
	return probabilites

'''La probabilidad condicional entonces puede llamarse por una nueva funcion que responda
¿cual es la probabilidad de que el segundo digito sea j dado a que el primer digito fue i?'''

def probCondicional(probConjunta_vector):
	
	1stDigit_linspace = np.arange(0, 10, 1)
	12Digits_linspace = np.arange(0, len(probConjunta_vector), 1)
	prob_msd = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6] #probabilidad del digito mas siginificativo por benford
	
	probCondicional = []
	for i in 1stDigit_linspace:
		for j in 12Digits_linspace:
			
			P = round(probConjunta_vector[j]/pro_msd[i], 2)
			probCondicional.append(P)
	return probCondicional

	
	



'''
def isConditional():
	
	#Por Benford las probabilidades para el 1er y 2do digito son:
	P_1stD = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
	P_2ndD = [12, 11.4, 10.9, 10.4, 10, 9.7, 9.3, 9, 8.8, 8.5]

	#Se pretende buscar la probalidad condicional 
	fst_symbols = np.arange(1, 10, 1)
	scnd_symbols = np.arange(0, 10, 1)

	for i in fst_symbols:
		for j in  scnd_symbols:
			
			P = 

'''


#========================MAIN==============================

file = str(input('Name a dataset from your file directory: '))
data = './database/{}'.format(file)
index = str(input('Input an index: '))

#CALLING FUNCTIONS
inputData = readData(data, indexName = index)
computeObserved = computeFstDigit(inputData)
computePredicted = lookForExpectedCounts(computeObserved[2])

#vars
ObsCounts = computeObserved[0]
ExpCounts = computePredicted[0]
ObsProbability = np.array(computeObserved[1])
BenProbability = np.array(computePredicted[1])

#call summary
printSummary(inputData, ObsCounts, ExpCounts, ObsProbability, BenProbability)







	


