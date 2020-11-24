#############################################################
#							    #
#      Estudio y Simulación de aplicaciones de la teoría    #
#	 de la probabilidad en la ingeniería eléctrica      #
#		   				            #
#	Jeaustin Sirias Chacon (jeaustin.sirias@ucr.ac.c)   #
#                     Copyright (C) 2020		    #
#							    #
#############################################################

# VARIABLES
TEST = ./test/
SOURCE = ./source/

# COMANDOS
require: # Install requirements
	pip install -r requirements.txt

run: # Run without installing
	jupyter notebook $(SOURCE)P1.ipynb \
	&& jupyter notebook $(SOURCE)P2.ipynb \
	&& jupyter notebook $(SOURCE)P3.ipynb \
	&& jupyter notebook $(SOURCE)P4.ipynb \


