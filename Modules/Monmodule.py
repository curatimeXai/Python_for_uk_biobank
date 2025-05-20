# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:52:20 2025

@author: paulmathieu
"""

import pandas as pd
import random as rd
import datetime
from matplotlib.ticker import LinearLocator
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import statistics

#the confusion matrix is a tool to check the accuracy and the stability of a regression model it shows differents statistics
def report_confusion_matrix(matrix,file):
    liste_cardio=['None of the above','High blood pressure','Stroke','Angina','Heart attack']
    i=0
    for k in matrix:
        file.write("\n"+liste_cardio[i]+"\n Predit HN   HP ")
        i+=1
        col=""
        matrix2=k
        for ligne in range(0,2):
            for column in range(0,2):
                col+=" "+str(matrix2[ligne,column])
            file.write("\n "+col)
            col=""
    

    

def statistic_report(model,df,liste,bias,liste2,categorie,matrix):
    chemin= str(datetime.datetime.now())
    chemin= str(datetime.datetime.now())
    chemin=chemin.replace(":","_")
    chemin="C:/Users/paulmathieu/Documents/Stage/etranger/dataset/BMI_exercice_VS_cardio/resultat_correlation/"+chemin+".csv"
    b=(df.loc[:,["Body mass index (BMI) | Instance 0","Body fat percentage | Instance 0"]]).corr()
    b.to_csv(chemin,sep=";")
    f=open(chemin,"a")
    for i in range(0,5):
        f.write("\n voila les resultats pour"+categorie[i])
        f.write("\n coefficient de regression : "+liste[i])
        f.write("\n biais de regression : "+bias[i])
        f.write("\n accuracy score : "+liste2[i])
    report_confusion_matrix(matrix,f)
    f.close()
    
    
def A_confusion_matrix(Xpred,Ypred,logr):
    A=logr.predict(Xpred)
    return confusion_matrix(A,Ypred)

def slicing_BMI(df):
    df1=df[df["Body mass index (BMI) | Instance 0"]<=18.5]
    df2=df[df["Body mass index (BMI) | Instance 0"]>=18.5]
    return df1,df2
