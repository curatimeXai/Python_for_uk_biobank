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
sys.path.append('./Modules')#here send the path where the module "Monmodule" is registered
import random as rd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import Monmodule
import statistics

path='./Bodyfat_BMI_Physical_activity.csv'#here must be the path of the data
df = pd.read_csv(path,sep=";")

print("pret")
"""ok=(df.loc[df["p20002_i0_a6"]=="heart/cardiac problem",['p21022']]).values
paok=(df.loc[df["p20002_i0_a6"]!="heart/cardiac problem",['p21022']]).values"""
#ici, probleme cardiaque selon cardio_location



def kind_diseases(dataf):
    a=dataf.groupby(["Vascular/heart problems diagnosed by doctor | Instance 0"], sort=False).sum()
    b=(a.iloc[:,1]).filter(regex=",",axis=0)
    return b
def filter_activity(df):
    liste=[]
    z=df[((df["Body mass index (BMI) | Instance 0"]).isna()==False) & ((df["Body fat percentage | Instance 0"]).isna()==False)]
    z=z[z["Vascular/heart problems diagnosed by doctor | Instance 0"]!="Prefer not to answer"]
    z=z[(z["Vascular/heart problems diagnosed by doctor | Instance 0"]).isna()==False]
    return z
cardio_list=['None of the above','High blood pressure','Stroke','Angina','Heart attack']

def correlation(data):
    z=data.loc[:,["Vascular/heart problems diagnosed by doctor | Instance 0","BMI+bodyfat","Body mass index (BMI) | Instance 0","Body fat percentage | Instance 0","BMIxbodyfat"]]
    cardio_list=['None of the above','High blood pressure','Stroke','Angina','Heart attack']
    #print("premachage",z["Vascular/heart problems diagnosed by doctor | Instance 0"])
    for k in cardio_list:
        z=data.loc[:,["Vascular/heart problems diagnosed by doctor | Instance 0","BMI+bodyfat","Body mass index (BMI) | Instance 0","Body fat percentage | Instance 0","BMIxbodyfat"]]
        z.loc[((z.loc[:,"Vascular/heart problems diagnosed by doctor | Instance 0"]).str.contains(k)==True),"Vascular/heart problems diagnosed by doctor | Instance 0"]=int(0)
        z.loc[((z["Vascular/heart problems diagnosed by doctor | Instance 0"])!=0),"Vascular/heart problems diagnosed by doctor | Instance 0"]=int(1)
        #z.loc[:,"Vascular/heart problems diagnosed by doctor | Instance 0"]=pd.to_numeric(z["Vascular/heart problems diagnosed by doctor | Instance 0"])
        #print("check etat 1",z[z["Vascular/heart problems diagnosed by doctor | Instance 0"]==0])
        #print("debut, k:",k,"\n",z.corr(method='pearson'),"\n")
        X=((z.loc[:,["Body fat percentage | Instance 0","Body mass index (BMI) | Instance 0"]]).to_numpy()).reshape(-1,2)
        Y=((z["Vascular/heart problems diagnosed by doctor | Instance 0"]).to_numpy())
        Y=Y.astype(int)
        print("X tail:",X.shape,"Y tail",Y.shape)
        logr = LogisticRegression()
        logr.fit(X[0:2],Y[0:2])
        print("score",logr.score(X[0:2],Y[0:2]),"taille",X[0:2].shape,"X")
        return X,Y,logr
          
#je me dois de passer à la covariance, les expositionsà la pollution ne sont pas identiquement  distribuéescela doit fausser le probleme
def create_colu(dat):
    dat["BMI+bodyfat"]=dat["Body fat percentage | Instance 0"]+dat["Body mass index (BMI) | Instance 0"]
    dat["BMIxbodyfat"]=dat["Body fat percentage | Instance 0"]*dat["Body mass index (BMI) | Instance 0"]
    dat["waist to height"]=dat["Waist circumference | Instance 0"]/dat["Standing height | Instance 0"]
    serie=dat["BMI+bodyfat"]
    print(serie.value_counts(ascending=True))
    return dat

def train_regression(z,CVD,column):
    c=z.copy()
    c.loc[((c.loc[:,"Vascular/heart problems diagnosed by doctor | Instance 0"]).str.contains(CVD)==True),"Vascular/heart problems diagnosed by doctor | Instance 0"]=int(0)
    c.loc[((c["Vascular/heart problems diagnosed by doctor | Instance 0"])!=0),"Vascular/heart problems diagnosed by doctor | Instance 0"]=int(1)
    #X=((c.loc[:,"Body fat percentage | Instance 0"]).to_numpy()).reshape(-1,1)#for the body fat
    X=((c.loc[:,column]).to_numpy()).reshape(-1,1)#for BMI+Bodyfat
    Y=((c["Vascular/heart problems diagnosed by doctor | Instance 0"]).to_numpy())
    Y=Y.astype(int)
    logr = LogisticRegression(penalty=None,solver="newton-cg")#(penalty=None,solver="newton-cg") if want to tests
    logr=logr.fit(X,Y)
    #regression polynomiale
    model = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression(fit_intercept=False))])
    # fit to an order-3 polynomial data
    polyn = model.fit(X, Y)
    polyn.named_steps['linear'].coef_
    return c,logr,X,Y,polyn,

test_tang2=[]
testliner=[]
z=filter_activity(df)
z=create_colu(z)
#X,Y,logr=correlation(z)
z1,z2=Monmodule.slicing_BMI(z)

def graphoc1(z): #this is the first graphoc
    #categories=np.arange(12.5,20,0.05)#for underweight people BMI
    #categories=np.arange(20,50,0.6) #for high and medium BMI
    #categories=np.arange(103.0368,1358,12.55)#for BMIxbodyfat
    #categories=np.arange(44.6,74.7,0.5)#for BMI+bodyfat
    categories=np.arange(0.443787,0.85,0.0034)#for waist to height
    #categories=np.arange(5,60,0.5) #for le bodyfat
    column="waist to height"
    #column="BMIxbodyfat"
    coeff_model_list=[]
    bias_modelèlist=[]
    mode_score_list=[]
    PbA_given_B=np.zeros(len(categories))
    logistic_regr=np.zeros(len(categories))
    kn1=0
    mse=[]
    matrix=[]
    #input("a")
    z=z[(z[column]).isna()==False]
    for u in cardio_list:
        #processing regression
        c,logr,X,Y,polynome=train_regression(z,u,column)
            #processing regression
        i=0
        for k in categories:
            #print(k,":k")
           # d=((z["BMIxbodyfat"]<=k) & (z["BMIxbodyfat"]>=k-12))# for BMIxBodyfat
            #d2=((z["BMIxbodyfat"]<=k) & (z["BMIxbodyfat"]>=k-12))# for BMIxBodyfat
            #d=((z["Body fat percentage | Instance 0"]<=k) & (z["Body fat percentage | Instance 0"]>=kn1)) #for body fat only
            #d=((z["BMI+bodyfat"]<=k) & (z["BMI+bodyfat"]>=k-0.5))# for BMI+Bodyfat
            #d2=((c["BMI+bodyfat"]<=k) & (c["BMI+bodyfat"]>=k-0.5))
            d=((z[column]<=k+0.001) & (z[column]>=k-0.001))# for waist to height
            #d2=((c[column]<=k+0.001) & (c[column]>=k-0.001))#for waist to height
            #d=(df["Body mass index (BMI) | Instance 0"]<=k+0.3) & (df["Body mass index (BMI) | Instance 0"]>k-0.3) #for BMI
            d2=d #for BMI
            kn1=k
            P_B=(z[d].shape)[0]/500000
            if(P_B==0):
                PbA_given_B[i]=(PbA_given_B[i-1]+PbA_given_B[i-2])/2
                logistic_regr[i]=(logr.predict_proba(np.array((k).reshape(-1,1))))[0,0]
                '''while P_B==0:
                    k=k*1.5
                    d=((z["BMIxbodyfat"]<=k+k/2) & (z["BMIxbodyfat"]>=k-k/2)) if you don't have any data it is good use to makes some means or series expansion to avoid irregular curves
                    P_B=(z[d].shape)[0]/500000'''
            #print("boucle b")
            else:
                internal_liste=np.zeros(5)
                b=c[d]
                #a=(z[d])["Vascular/heart problems diagnosed by doctor | Instance 0"]#we transform the data into series, the number of people with the symptom would be calculated with the number of rows
                a=(z[d])["Vascular/heart problems diagnosed by doctor | Instance 0"]
                taille=((a.loc[(a.str.contains(u)==True)]).shape)[0]
                PA_inter_B=taille/500000
                PbA_given_B[i]=PA_inter_B/P_B
                '''
                #MSE
                y=((c[d2]).loc[:,"Vascular/heart problems diagnosed by doctor | Instance 0"]).to_numpy()
                x=(((c[d2]).loc[:,column]).to_numpy()).reshape(-1,1)
                y_pred=logr.predict(x)
                #print(y,y_pred)
                mse+=[mean_squared_error(y,y_pred)]
                #MSE
                '''
                #m=((b.loc[:,"Body fat percentage | Instance 0"]).to_numpy())
                #m=((b.loc[:,"BMIxbodyfat"]).to_numpy())
                m=((b.loc[:,column]).to_numpy())
                logistic_regr[i]=(logr.predict_proba(np.array((np.mean(m)).reshape(-1,1))))[0,0] #this is the probability calculated from the logistic regression
            i+=1
        '''
        variance=statistics.variance(mse[:])
        median=statistics.median(mse[:])'''
        plt.plot(categories,PbA_given_B,color = 'g')#for probability from Bayes theorem (to be avoid when not enough data)
        plt.plot(categories,logistic_regr,color = 'r')#For Probability from regression (can work even if there is no case in the data)
        plt.xlabel(column)
        plt.ylabel(u)
        plt.show()#the matplotlib is used for showing graphics with plt.
        mode_score_list+=[str(logr.score(X,Y))]
        coeff_model_list+=[str(logr.coef_)]
        bias_modelèlist+=[str(logr.intercept_)]
        matrix+=[Monmodule.A_confusion_matrix(X,Y,logr)]#for the  prediction matrix
        print("quantiles:",z["waist to height"].quantile([0.1,0.5,0.99]))
        #print("loop c,")
    #input("end")
    Monmodule.statistic_report(logr,z,coeff_model_list,bias_modelèlist,mode_score_list,cardio_list,matrix)
graphoc1(z2)

def prediction(categories,PbA_given_B):
    x=np.array([categories[18:55]]).reshape(-1,1)
    y=np.array([PbA_given_B[18:55]]).reshape(-1,1)
    '''X=np.zeros((1,22))
    Y=np.zeros((1,22))'''
    print("entree")
    print("sortie")
    reg = LinearRegression().fit(x, y)
    reg.score(x,y)
    print('coef',reg.coef_)#coefficients)
    print("intercept",reg.intercept_)#bias
    z=reg.predict(np.array([categories[:]]).reshape(-1,1))


def cube_dependance_BMI_bodyfat(z):
    #categories_BMI=np.arange(16.1918,52.5,0.5)
    #categories=np.arange(5,60,1)
    #categories_BMI=np.arange(17,52,1)
    categories=np.arange(5,60,2)
    categories_BMI=np.arange(17,52,2)
    PbA_given_B=np.zeros([len(categories),len(categories_BMI)])
    logistic_regr=np.zeros([len(categories),len(categories_BMI)])
    Z=0
    #input("a")
    u="High blood pressure"
        #calcul regre logistique
    c=z.copy()
    c.loc[((c.loc[:,"Vascular/heart problems diagnosed by doctor | Instance 0"]).str.contains(u)==True),"Vascular/heart problems diagnosed by doctor | Instance 0"]=int(0)
    c.loc[((c["Vascular/heart problems diagnosed by doctor | Instance 0"])!=0),"Vascular/heart problems diagnosed by doctor | Instance 0"]=int(1)
    X=((c.loc[:,"Body fat percentage | Instance 0"]).to_numpy()).reshape(-1,1)
    Y=((c.loc[:,"Vascular/heart problems diagnosed by doctor | Instance 0"]).to_numpy())
    Y=Y.astype(int)
    logr = LogisticRegression()
    logr.fit(X,Y)
            #calcul regre logistique
    for k2 in range(0,len(categories_BMI)):
        #BMI must be the column of PA sachant B and the x coordinate of the graph
        #body has to be the rows of PA sachant B and the coordinate y PA_B([y,x])
        i=0
        z2=z[(z["Body mass index (BMI) | Instance 0"]<=categories_BMI[k2]+1) & (z["Body mass index (BMI) | Instance 0"]>categories_BMI[k2]-1)]
        kn1=0
        for k in categories:
            #print(k,":k")
            #d=((z["BMIxbodyfat"]<=k+17) & (z["BMIxbodyfat"]>=k-17))
            
            d=((z2["Body fat percentage | Instance 0"]<=k) & (z2["Body fat percentage | Instance 0"]>=kn1))
            kn1=k
            P_B=(z2[d].shape)[0]/500000
            if(P_B==0):
                #PbA_given_B[i,k2]=0
                PbA_given_B[i,k2]=(PbA_given_B[i-1,k2]+PbA_given_B[i,k2-1])/2#looking for better way to 
                '''while P_B==0:
                    k=k*1.5
                    d=((z["BMIxbodyfat"]<=k+k/2) & (z["BMIxbodyfat"]>=k-k/2))
                    P_B=(z[d].shape)[0]/500000'''
            #print("boucle b")
            else:
                internal_liste=np.zeros(5)
                #b=c[d]
                #a=(z[d])["Vascular/heart problems diagnosed by doctor | Instance 0"]#on réduit à l'état de série symptome le dataframe "on a besoin que des expressions des symptomes", la quantité d'individus sera compté à partir des lignes
                a=(z2[d])["Vascular/heart problems diagnosed by doctor | Instance 0"]
                taille=((a.loc[(a.str.contains(u)==True)]).shape)[0]
                PA_inter_B=taille/500000
                PbA_given_B[i,k2]=PA_inter_B/P_B
                m=((z2.loc[d,["Body fat percentage | Instance 0"]])).to_numpy()
                m2=((z2.loc[d,["Body mass index (BMI) | Instance 0"]]).to_numpy())
                logistic_regr[i,k2]=(logr.predict_proba((np.array(([np.mean(m),np.mean(m2)])).reshape(2,1))))[0,0]
                Z = np.sin(categories_BMI[k2])
            i+=1
    
    X=((z.loc[0:2,["Body fat percentage | Instance 0","Body mass index (BMI) | Instance 0"]])).to_numpy().reshape(-1,2)
    Z=((z.loc[0:,"Vascular/heart problems diagnosed by doctor | Instance 0"]).to_numpy())
    Axex=categories_BMI
    Axey=categories
    #Axez=np.arange(0,1,0.01)
    print(X,"X et Y")
    Axex,Axey=np.meshgrid(Axex,Axey)
        
    fig=plt.figure()
    ax=fig.add_subplot(projection='3d')
    #ax.plot_surface(Axex,Axey,PbA_given_B.reshape(73,110), cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.plot_surface(Axex,Axey,PbA_given_B, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    #ax.plot_surface(Axex,Axey,logistic_regr.reshape(len(categories_BMI),len(categories)), cmap=cm.coolwarm,linewidth=0, antialiased=False)
    #ax.plot_surface(Axex,Axey,logistic_regr.reshape(73,110), cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.set_xlabel('body fat')
    ax.set_ylabel('BMI')
    ax.set_zlabel('Z Label')
    plt.show()
    print("boucle c,")
    return PbA_given_B
rec=cube_dependance_BMI_bodyfat(z)



#on en conclue que l'on a pu approximer le systeme à 0.02274089*x+-0.3255023
