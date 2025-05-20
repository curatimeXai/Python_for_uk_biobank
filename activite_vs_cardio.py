# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:01:34 2025

@author: paulmathieu
"""



import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

path="./intense_activity_via_Duration_of_heavy_DIY.csv"
df = pd.read_csv(path,sep=";")
print("pret")
"""ok=(df.loc[df["p20002_i0_a6"]=="heart/cardiac problem",['p21022']]).values
paok=(df.loc[df["p20002_i0_a6"]!="heart/cardiac problem",['p21022']]).values"""
#ici, probleme cardiaque selon cardio_location

def creer_col(dat):
    dat["nouvel indice"]=dat.loc[:,"Duration of moderate activity | Instance 0"]*dat.loc[:,"Number of days/week of moderate physical activity 10+ minutes | Instance 0"]
    serie=dat["nouvel indice"]
    print(serie.value_counts(ascending=True))
    return dat
#categories=np.arange(16.1918,52.5,1.2)
liste_cardio=['None of the above','High blood pressure','Stroke','Angina','Heart attack']
liste_dur_cardio=["Less than 15 minutes","Between 15 and 30 minutes","Between 30 minutes and 1 hour","Between 1 and 1.5 hours","Between 1.5 and 2 hours","Between 2 and 3 hours","Over 3 hours"]
liste_dur_cardio_graph=["8","28", "45","75","105","150",">150"]
def filtre_maladies(dataf):
    a=dataf.groupby(["Vascular/heart problems diagnosed by doctor | Instance 0"], sort=False).sum()
    b=(a.iloc[:,1]).filter(regex=",",axis=0)
    return b
def filtre_activité(df):
    liste=[]
    #z=df[df["Duration of other exercises | Instance 0"].isna()==False] inutile car le sort==false filtre déjà
    z=df[(df["Duration of moderate activity | Instance 0"]!='Do not know') | (df["Duration of moderate activity | Instance 0"]!='Prefer not to answer')]
    z=z.groupby(["Duration of moderate activity | Instance 0"], sort=False).sum()
    for i in z.index.values:
        if(i=="Prefer not to answer" or i=="Do not know"):
            liste+=[-2]
        else:
            liste+=[int(i)]
    return liste
'''a=filtre_activité(df)
a.sort()
a=a[2:]'''

def filtre_activ2(df):
    z=df[(df["Duration of moderate activity | Instance 0"]!='Do not know') & (df["Duration of moderate activity | Instance 0"]!='Prefer not to answer')]
    z=z[(z["Number of days/week of moderate physical activity 10+ minutes | Instance 0"]!="Do not know") & (z["Number of days/week of moderate physical activity 10+ minutes | Instance 0"]!="Prefer not to answer")]
    z=z[z["Duration of moderate activity | Instance 0"].isna()==False]
    z=z[z["Number of days/week of moderate physical activity 10+ minutes | Instance 0"].isna()==False]
    #z=z[z["Duration of moderate activity | Instance 0"].isna()==True]["Duration of moderate activity | Instance 0"]="0"
    #z.loc[z["Duration of moderate activity | Instance 0"].isna(), "Duration of moderate activity | Instance 0"] = "0"
    z.loc[z["Duration of moderate activity | Instance 0"]=='None of the above',"Duration of moderate activity | Instance 0"] = "0"
    #z.loc[z["Number of days/week of moderate physical activity 10+ minutes | Instance 0"].isna(), "Number of days/week of moderate physical activity 10+ minutes | Instance 0"] = "0"
    z.loc[z["Number of days/week of moderate physical activity 10+ minutes | Instance 0"]=='None of the above',"Number of days/week of moderate physical activity 10+ minutes | Instance 0"] = "0"
    z["Duration of moderate activity | Instance 0"].convert_dtypes(convert_integer=True)
    z.loc[:,"Duration of moderate activity | Instance 0"]=pd.to_numeric(z["Duration of moderate activity | Instance 0"])
    z["Number of days/week of moderate physical activity 10+ minutes | Instance 0"].convert_dtypes(convert_integer=True)
    z.loc[:,"Number of days/week of moderate physical activity 10+ minutes | Instance 0"]=pd.to_numeric(z["Number of days/week of moderate physical activity 10+ minutes | Instance 0"])
    #print(z)
    #z=z.groupby(["Duration of moderate activity | Instance 0"], sort=False).sum()
    #z=z.sort_index(level=int)
    return z


#si je souhaite compter tout les cas correspondants à une valeur de symptome CVD je dois faire groupby("Vascular/heart problems diagnosed by doctor | Instance 0").value_counts(ascending=True)
#si je souhaite connaître la répartition d'une ordonnée, je n'ai qu'à faire (dataframe["nouvel indice"]).quantile(dec) où dec représente les différents quantiles [0.1,0.2,0.5,0.75,0.9]
def calcul_moy(data):
    z=data[data["Duration of moderate activity | Instance 0"].isna()==False]
    z=z[(z["Duration of moderate activity | Instance 0"]!='Do not know') & (z["Duration of moderate activity | Instance 0"]!='Prefer not to answer')]
    z.loc[z["Vascular/heart problems diagnosed by doctor | Instance 0"]!='Heart attack',"Vascular/heart problems diagnosed by doctor | Instance 0"]=0
    z.loc[z["Vascular/heart problems diagnosed by doctor | Instance 0"]=='Heart attack',"Vascular/heart problems diagnosed by doctor | Instance 0"]=1
    condi2=((z["Vascular/heart problems diagnosed by doctor | Instance 0"]==0))
    condi1=((z["Vascular/heart problems diagnosed by doctor | Instance 0"]==1))
    analyse_cov=z.loc[:,["Vascular/heart problems diagnosed by doctor | Instance 0","Duration of moderate activity | Instance 0"]]
    moy1_malade=((z[(condi1)]).loc[:,"Duration of moderate activity | Instance 0"]).mean()
    moy2_sain=((z[(condi2)]).loc[:,"Duration of moderate activity | Instance 0"]).mean()
    moy3=(z.loc[:,"Duration of moderate activity | Instance 0"]).mean()
    print(moy1_malade,"malad",moy2_sain,"sain",moy3,"moyen","\n",analyse_cov.corr(method='pearson'),"corr","vasc")
    return z

def balayage(l,data):
    malist=[]
    parcour=(data.groupby(["Vascular/heart problems diagnosed by doctor | Instance 0"], sort=False).sum())
    parcour=parcour.index.to_list()
    for k in parcour:
        if(k.find(l)!=-1):
            #print(l,"l",k,"k")
            malist+=[k]
    return malist
        
quid=filtre_activ2(df)
di=calcul_moy(quid)
datplus=creer_col(quid)
#il y a environ 255 valeurs de duration of moderate différents s'étalant de 0 à 1440. 
#soit je tente de calculer le pas moyen avec pas moy = (somme (poid actuel-poidprec))/255
#soit je tente un pas de 144 pour avoir 100 pas... peut être tenter un pas de 75 ?

#note la médiane est à 40, le dernier quartile est à 60
l=filtre_activ2(df)
l2=creer_col(l)
def mesures2(z):
    x2=np.linspace(0,720,20)
    #z=data[data["nouvel indice"].isna()==False]
    z=z[(z["nouvel indice"]!='Do not know') & (z["nouvel indice"]!='Prefer not to answer')]
    PA_sachant_B=np.zeros((20))  
    PA_inter_B=0
    for k in liste_cardio:
        liste=balayage(k,z)
        for i in range(1,20):
            condi1=((z["nouvel indice"]>=x2[i-1]) & (z["nouvel indice"]<=x2[i]))
            #condi2=((z["Vascular/heart problems diagnosed by doctor | Instance 0"]!='None of the above') & (z["Vascular/heart problems diagnosed by doctor | Instance 0"].isna()==False))
            for j in liste:
                condi2=((z["Vascular/heart problems diagnosed by doctor | Instance 0"]==j))
                PA_inter_B+=(z[(condi2) & (condi1)]).shape[0]#/z.shape[0]  
            P_B=(z[condi1]).shape[0]#/z.shape[0] s'annule dans la division
            PA_sachant_B[i]=PA_inter_B/P_B
            PA_inter_B=0
        plt.plot(x2,PA_sachant_B)
        plt.xlabel("moderate activity")
        plt.ylabel(k)
        plt.show()
    
    return [PA_sachant_B]
#x=mesures2(l2)
       
     
                       
def mesures(l1,a):
    print("nouveau graph")
    z=df[df["Duration of moderate activity | Instance 0"].isna()==False]
    z=z[(z["Duration of moderate activity | Instance 0"]!='Do not know') | (z["Duration of moderate activity | Instance 0"]!='Prefer not to answer')]
    PA_sachant_B=np.zeros((len(a)))
    P2A_sachant_B=np.zeros((int(len(a)/3)+1))
    x=np.zeros((int(len(a)/3)+1))
    i=1
    t=0
    moy_par3=1
    for k in a[3:]:
        d=(z["Duration of moderate activity | Instance 0"]==str(k))
        neo_z=z[d]#df qui vérifie la condition z (la valeur)
        P_B=(z[d].shape)[0]/(z.shape[0])#la longueur de z
        internal_liste=np.zeros(5)
        taille_a_inter_b=(neo_z[(neo_z["Vascular/heart problems diagnosed by doctor | Instance 0"]!='None of the above') & (neo_z["Vascular/heart problems diagnosed by doctor | Instance 0"].isna()==False)]).shape[0]
        PA_inter_B=taille_a_inter_b/(z.shape[0])
        PA_sachant_B[i-1]=PA_inter_B/P_B
        t+=k
        if(moy_par3==3):
            P2A_sachant_B[int(i/3)]=(PA_sachant_B[i-1]+PA_sachant_B[i-2]+PA_sachant_B[i-3])/3
            x[int(i/3)]=t/3
            moy_par3=0
            t=0
        i+=1
        moy_par3+=1
    #plt.plot(a,PA_sachant_B)
    plt.plot(x,P2A_sachant_B)
    plt.xlabel("activity")
    plt.ylabel("heart diseseases")
    plt.show()
    return [PA_sachant_B]

#ad=mesures2(l)
''' 
        

def mesures(l1,a):
    print("nouveau graph")
    z=df[df["Duration of moderate activity | Instance 0"].isna()==False]
    z=z[(z["Duration of moderate activity | Instance 0"]!='Do not know') | (z["Duration of moderate activity | Instance 0"]!='Prefer not to answer')]
    PA_sachant_B=np.zeros((len(a),5))
    i=0
    for k in a[3:]:
        d=(df["Duration of moderate activity | Instance 0"]==str(k))
        print(z[d],"z vérifiant que la durée est égale à k")
        P_B=(z[d].shape)[0]/(z.shape[0])#la longueur de z
        print("boucle b")
        internal_liste=np.zeros(5)
        for u in range(0,5):
            j=liste_cardio[u]
            print("j:",j)
            #PA_inter_B=(((df[((df["Vascular/heart problems diagnosed by doctor | Instance 0"]==j) & (df["Vascular/heart problems diagnosed by doctor | Instance 0"].isna()==False)) & (d)]).shape)[0])/500000
            tout_B=(z[d])["Vascular/heart problems diagnosed by doctor | Instance 0"]#on réduit à l'état de série symptome le dataframe "on a besoin que des expressions des symptomes", la quantité d'individus sera compté à partir des lignes
            taille_B_verifie_a=((tout_B.loc[(tout_B.str.contains(j,regex=True)==True)]).shape)[0]
            PA_inter_B=taille_B_verifie_a/(z.shape[0])
            internal_liste[u]=PA_inter_B/P_B
        PA_sachant_B[i]=internal_liste
        i+=1
    print(tout_B.shape,len(PA_sachant_B),"a et pa sachant b")
    for l in range(0,5):
        plt.plot(a,PA_sachant_B[:,l])
        plt.xlabel("activity")
        plt.ylabel(liste_cardio[l])
        plt.show()

mesures(a,a)
'''


def historigram(data,a,duration,symptom):
    print("nouveau graph")
    z=data[data["Duration of heavy DIY | Instance 0"].isna()==False]
    z=z[(z["Duration of heavy DIY | Instance 0"]!='Do not know') | (z["Duration of heavy DIY | Instance 0"]!='Prefer not to answer')]
    PA_sachant_B=np.zeros((len(duration)))
    k=0
    PB=0
    PA_inter_B=0
    liste_cvd=symptom[1:]
    for l in liste_cvd:
        #list2=(balayage(l,z))[1:] je n'en ai plus besoin grâce à str.contains
        #print(list2,l, "separateur \n")
        for i in duration:
            #print(list2,"liste 2","l",l)
            #indépendant du symptome, il vaut mieux la sortir pour ne pas alourdir la boucle
            condib=(z["Duration of heavy DIY | Instance 0"]==i)
            PB+=(z[condib]).shape[0]
            #indépendant du symptome, il vaut mieux la sortir pour ne pas alourdir la boucle
            condia_et_b=((z["Duration of heavy DIY | Instance 0"]==i) & ((z["Vascular/heart problems diagnosed by doctor | Instance 0"]).str.contains(l)==True))
            PA_inter_B+=(z[condia_et_b]).shape[0]
            PA_sachant_B[k]=PA_inter_B/PB
            PB=0
            PA_inter_B=0
            k+=1
        plt.bar(a,PA_sachant_B)
        plt.xlabel("intense activity")
        plt.ylabel(l)
        plt.show()
        k=0
    
historigram(l, liste_dur_cardio_graph,liste_dur_cardio,liste_cardio)