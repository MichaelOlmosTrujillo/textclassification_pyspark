# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 08:34:50 2022

@author: miolmos
"""
import pandas as pd
from pyspark import SparkContext

clase_positiva_df = pd.read_csv('proyectos_clase_positiva.csv', 
                                sep = ';')
print(clase_positiva_df.head())
clase_negativa_df = pd.read_csv('proyectos_clase_negativa.csv',
                                sep = ';')
print(clase_negativa_df.head())
print(clase_positiva_df.shape)
print(clase_negativa_df.shape)
print(clase_positiva_df.info())
print(clase_negativa_df.info())
print(clase_positiva_df['codBPIN'].tail())
print(clase_negativa_df['codBPIN'].tail())
print(clase_positiva_df['nomProy'][1])

