# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:05:12 2025

@author: paulmathieu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:02:06 2025

@author: paulmathieu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:51:14 2025

@author: paulmathieu
"""


import pandas as pd
import sys
import random as rd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
import Monmodule
sys.path.append('C:/Users/paulmathieu/Documents/Stage/etranger/dataset/BMI_exercice_VS_cardio/Modules/')
import datetime




#df = pd.read_csv('C:/Users/paulmathieu/Documents/Stage/etranger/dataset/cardiovascular_location.csv')
df = pd.read_csv('C:/Users/paulmathieu/Documents/Stage/etranger/dataset/BMI_exercice_VS_cardio/BMI_exercises_bodyfat_vs_cardio.csv',sep=";")

print("pret")
"""ok=(df.loc[df["p20002_i0_a6"]=="heart/cardiac problem",['p21022']]).values
paok=(df.loc[df["p20002_i0_a6"]!="heart/cardiac problem",['p21022']]).values"""
#ici, probleme cardiaque selon cardio_location



def filtre_maladies(dataf):
    a=dataf.groupby(["Vascular/heart problems diagnosed by doctor | Instance 0"], sort=False).sum()
    b=(a.iloc[:,1]).filter(regex=",",axis=0)
    return b
def filtre_activité(df):
    liste=[]
    #z=df[df["Duration of other exercises | Instance 0"].isna()==False] inutile car le sort==false filtre déjà
    z=df[((df["Body mass index (BMI) | Instance 0"]).isna()==False) & ((df["Body fat percentage | Instance 0"]).isna()==False)]
    z=z[z["Vascular/heart problems diagnosed by doctor | Instance 0"]!="Prefer not to answer"]
    z=z[(z["Vascular/heart problems diagnosed by doctor | Instance 0"]).isna()==False]
    return z
liste_cardio=['None of the above','High blood pressure','Stroke','Angina','Heart attack']

def correlation(data):
    z=data.loc[:,["Vascular/heart problems diagnosed by doctor | Instance 0","BMI+bodyfat","Body mass index (BMI) | Instance 0","Body fat percentage | Instance 0","BMIxbodyfat"]]
    liste_cardio=['None of the above','High blood pressure','Stroke','Angina','Heart attack']
    #print("premachage",z["Vascular/heart problems diagnosed by doctor | Instance 0"])
    for k in liste_cardio:
        z=data.loc[:,["Vascular/heart problems diagnosed by doctor | Instance 0","BMI+bodyfat","Body mass index (BMI) | Instance 0","Body fat percentage | Instance 0","BMIxbodyfat"]]
        z.loc[((z.loc[:,"Vascular/heart problems diagnosed by doctor | Instance 0"]).str.contains(k)==True),"Vascular/heart problems diagnosed by doctor | Instance 0"]=int(0)
        z.loc[((z["Vascular/heart problems diagnosed by doctor | Instance 0"])!=0),"Vascular/heart problems diagnosed by doctor | Instance 0"]=int(1)
        #z.loc[:,"Vascular/heart problems diagnosed by doctor | Instance 0"]=pd.to_numeric(z["Vascular/heart problems diagnosed by doctor | Instance 0"])
        #print("check etat 1",z[z["Vascular/heart problems diagnosed by doctor | Instance 0"]==0])
        #print("debut, k:",k,"\n",z.corr(method='pearson'),"\n")
        X=((z.loc[:,"Body fat percentage | Instance 0"]).to_numpy()).reshape(-1,1)
        Y=((z["Vascular/heart problems diagnosed by doctor | Instance 0"]).to_numpy())
        Y=Y.astype(int)
        print("X tail:",X.shape,"Y tail",Y.shape)
        logr = LogisticRegression()
        logr.fit(X,Y)
        print("score",logr.score(X,Y))
        return X,Y,logr
          
#je me dois de passer à la covariance, les expositionsà la pollution ne sont pas identiquement  distribuéescela doit fausser le probleme
def creer_col(dat):
    dat["BMI+bodyfat"]=dat["Body fat percentage | Instance 0"]+dat["Body mass index (BMI) | Instance 0"]
    dat["BMIxbodyfat"]=dat["Body fat percentage | Instance 0"]*dat["Body mass index (BMI) | Instance 0"]
    serie=dat["BMIxbodyfat"]
    print(serie.value_counts(ascending=True))
    return dat


test_tang2=[]
testliner=[]
z=filtre_activité(df)
z=creer_col(z)
X,Y,logr=correlation(z)

#categories=np.arange(103.0368,1358,12.55) #pour BMIxbodyfat
categories=np.arange(44.52667,74.54451,0.6)#for BMI+bodyfat
#categories=np.arange(5,60,0.5) #pour body fat simple
PA_sachant_B=np.zeros(len(categories))
logistic_regr=np.zeros(len(categories))
model_coeff_liste=[]
mode_score_liste=[]
kn1=0
#input("a")
for u in liste_cardio:
    #calcul regre logistique
    c=z.copy()
    c.loc[((c.loc[:,"Vascular/heart problems diagnosed by doctor | Instance 0"]).str.contains(u)==True),"Vascular/heart problems diagnosed by doctor | Instance 0"]=int(0)
    c.loc[((c["Vascular/heart problems diagnosed by doctor | Instance 0"])!=0),"Vascular/heart problems diagnosed by doctor | Instance 0"]=int(1)
    #X=((c.loc[:,"Body fat percentage | Instance 0"]).to_numpy()).reshape(-1,1) #pour bodyfat
    X=((c.loc[:,"BMI+bodyfat"]).to_numpy()).reshape(-1,1)
    Y=((c["Vascular/heart problems diagnosed by doctor | Instance 0"]).to_numpy())
    Y=Y.astype(int)
    logr = LogisticRegression()
    logr.fit(X,Y)
        #calcul regre logistique
    i=0
    for k in categories:
        #print(k,":k")
        #d=((z["BMIxbodyfat"]<=k+17) & (z["BMIxbodyfat"]>=k-17))
        d=((z["BMI+bodyfat"]<=k+0.3) & (z["BMI+bodyfat"]>=k-0.3))#pour Bodyfat+BMI
        #d=((z["Body fat percentage | Instance 0"]<=k) & (z["Body fat percentage | Instance 0"]>=kn1))
        kn1=k
        P_B=(z[d].shape)[0]/500000
        if(P_B==0):
            PA_sachant_B[i]=1
            '''while P_B==0:
                k=k*1.5
                d=((z["BMIxbodyfat"]<=k+k/2) & (z["BMIxbodyfat"]>=k-k/2))
                P_B=(z[d].shape)[0]/500000'''
        #print("boucle b")
        else:
            internal_liste=np.zeros(5)
            b=c[d]
            #a=(z[d])["Vascular/heart problems diagnosed by doctor | Instance 0"]#on réduit à l'état de série symptome le dataframe "on a besoin que des expressions des symptomes", la quantité d'individus sera compté à partir des lignes
            a=(z[d])["Vascular/heart problems diagnosed by doctor | Instance 0"]
            taille=((a.loc[(a.str.contains(u)==True)]).shape)[0]
            PA_inter_B=taille/500000
            PA_sachant_B[i]=PA_inter_B/P_B
            m=((b.loc[:,"BMI+bodyfat"]).to_numpy())
            logistic_regr[i]=(logr.predict_proba(np.array((np.mean(m)).reshape(-1,1))))[0,0]
        i+=1
    plt.plot(categories,PA_sachant_B,color = 'g')
    plt.plot(categories,logistic_regr,color = 'r')
    plt.xlabel("BMI+bodyfat")
    plt.ylabel(u)
    plt.show()
    print("boucle c,")
    mode_score_liste+=[str(logr.score(X,Y))]
    model_coeff_liste+=[str(logr.coef_)]
#input("fin")
Monmodule.statistiques_report(logr,z,model_coeff_liste,mode_score_liste,liste_cardio)
def prediction(categories,PA_sachant_B):
    x=np.array([categories[18:55]]).reshape(-1,1)
    y=np.array([PA_sachant_B[18:55]]).reshape(-1,1)
    '''X=np.zeros((1,22))
    Y=np.zeros((1,22))'''
    print("entree")
    print("sortie")
    reg = LinearRegression().fit(x, y)
    reg.score(x,y)
    print('coef',reg.coef_)#coefficients)
    print("intercept",reg.intercept_)#biais
    z=reg.predict(np.array([categories[:]]).reshape(-1,1))

def graphic(categories,PA_sachant_B,liste_cardio):
    for i in range(0,5):
        #plt.plot(categories,z)
        plt.plot(categories,PA_sachant_B[:,i])
        #plt.plot(categories,test_log)
        plt.xlabel("BMI")
        plt.ylabel(liste_cardio[i])
        plt.show()
        #modele valable de 8 à 21
#graphic(categories, PA_sachant_B, liste_cardio)    
def graphic2(categories,PA_sachant_B):
    for i in range(0,3):
        #plt.plot(categories,z)
        plt.plot(categories,PA_sachant_B[:,i])
        plt.plot(categories,testliner)
            #plt.plot(categories,test_tang2)
            #plt.plot(categories,test_log)
        plt.xlabel("BMI*bodyfat")
        plt.ylabel("probability")
        plt.show()

        #modele valable de 8 à 21
def grocub_mesures():
    print("nouveau graph")
    cat_exerc=filtre_activité(df)
    z=df[df["Duration of other exercises | Instance 0"].isna()==False]
    longueur_activit=len(cat_exerc[0])
    categories=np.arange(16.1918,52.5,0.5)
    PA_sachant_B=np.zeros((len(categories),5))
    i=0
    for k in categories:
        testliner+=[k*0.02160812-0.27672888] #linéaire centré autour de (35,0.45)
        test_tang2+=[(np.tanh(k*0.02160812-0.27672888))] #tangeante centré autour de (35,0.45)
        d=(z["Body mass index (BMI) | Instance 0"]<=k+0.5) & (z["Body mass index (BMI) | Instance 0"]>k-0.5)
        P_B=(z[d].shape)[0]/235967#la longueur de z
        print("boucle b")
        internal_liste=np.zeros(5)
        for u in range(0,5):
            j=liste_cardio[u]
            print("j:",j)
            #PA_inter_B=(((df[((df["Vascular/heart problems diagnosed by doctor | Instance 0"]==j) & (df["Vascular/heart problems diagnosed by doctor | Instance 0"].isna()==False)) & (d)]).shape)[0])/500000
            a=(z[d])["Vascular/heart problems diagnosed by doctor | Instance 0"]#on réduit à l'état de série symptome le dataframe "on a besoin que des expressions des symptomes", la quantité d'individus sera compté à partir des lignes
            taille=((a.loc[(a.str.contains(j,regex=True)==True)]).shape)[0]
            PA_inter_B=taille/235967
            internal_liste[u]=PA_inter_B/P_B
        PA_sachant_B[i]=internal_liste
        i+=1
        print("boucle c,")

#on en conclue que l'on a pu approximer le systeme à 0.02274089*x+-0.3255023

def statistiques_report(model,df,liste):
    chemin= str(datetime.datetime.now())
    chemin=chemin.replace(":","_")
    chemin="C:/Users/paulmathieu/Documents/Stage/etranger/dataset/BMI_exercice_VS_cardio/resultat_correlation/"+chemin+".csv"
    b=(df.loc[:,["Body mass index (BMI) | Instance 0","Body fat percentage | Instance 0"]]).corr()
    b.to_csv(chemin,sep="   ")
    f=open(chemin+".csv","w")
    for i in range((0,3)):
        f.write("\n coefficient de regression : "+liste[i])
    f.close()
