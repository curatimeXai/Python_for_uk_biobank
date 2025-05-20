# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:51:14 2025

@author: paulmathieu
"""


import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import numpy as np


path="./income_vs_cardio.csv"#choose your path
df = pd.read_csv(path,sep=",")

print("pret")
"""ok=(df.loc[df["p20002_i0_a6"]=="heart/cardiac problem",['p21022']]).values
paok=(df.loc[df["p20002_i0_a6"]!="heart/cardiac problem",['p21022']]).values"""
#ici, probleme cardiaque selon cardio_location

categories=["18k<","18k->31k","31k->51k_","51k->100k","<100k","unknow ?"]
filtre=["Less than 18,000","18,000 to 30,999","31,000 to 51,999","52,000 to 100,000","Greater than 100,000","Prefer not to answer"]
liste_cardio=['None of the above','High blood pressure','Stroke','Angina','Heart attack']

PA_sachant_B=[]
for y in liste_cardio:
    for k in filtre:
        print("boucle")
        P_B=(df[df["Average total household income before tax | Instance 0"]==k].shape)[0]/30000
        print("boucle b")
        PA_inter_B=(((df[(df["Vascular/heart problems diagnosed by doctor | Instance 0"].str.contains(y)==True) & (df["Average total household income before tax | Instance 0"]==k)]).shape)[0])/30000
        print("boucle c,",PA_inter_B)
        PA_sachant_B+=[PA_inter_B/P_B]

    print("extract","pb",PA_sachant_B)
    
    
    
    print("ok jusqu'ici")
    #a=np.random.rand(3)
    
    plt.bar(categories,PA_sachant_B)
    plt.xlabel("incomes")
    plt.ylabel(y+" probability")
    plt.show()
    PA_sachant_B=[]

#je me dois de passer à la covariance, les expositionsà la pollution ne sont pas identiquement  distribuéescela doit fausser le probleme

moy=0
for i in range(0,6):
    print(i)
    moy=moy+PA_sachant_B[i]
moy=moy/5
    