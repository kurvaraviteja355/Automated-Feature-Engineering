import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import RFECV
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from boruta import BorutaPy
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from keras.models import Model
from keras.layers import Input, Dense
    
 


def create(X, X_column_types, y, y_column_types, arm, **kwargs):
    method = kwargs.get("method", "RFE_rf")
    method = kwargs.get("method", "RFE_Lasso")
    method = kwargs.get("method", "Lasso")
    method = kwargs.get("method", "Boruta")
    method = kwargs.get("method", "Autoencoder")
    method = kwargs.get("method", "Boruta")
    method = kwargs.get("method", "XBG")    
    # finding distance correlation b/w the features
    def distcorr(X, Y):
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        if np.prod(X.shape) == len(X):
            X = X[:, None]
        if np.prod(Y.shape) == len(Y):
            Y = Y[:, None]
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        n = X.shape[0]
        if Y.shape[0] != X.shape[0]:
            raise ValueError('Number of samples must match')
        a = squareform(pdist(X))
        b = squareform(pdist(Y))
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        return dcor

    ############# Shuffle the data ###################
    X['target'] = y
    data = shuffle(X)
    X = data.drop(['target'], axis=1)
    y = data['target']
    
    # AutoLearn  will take only the numerical features 
    
    num_X = X.select_dtypes(include = [int, float])

    #################### preprocessing step ###################
    if (y_column_types == "int64" or y_column_types == "float64"): # Regression 
    #if method == "mutual_information":
        c,d = num_X.shape
        names = np.arange(d)
        mode=SelectKBest(mutual_info_regression, k='all')
        mode.fit_transform(num_X,y)

        feats=sorted(zip(map(lambda c: round(c, 4), mode.scores_), names), reverse=True)

        finale=[]
        for i in range(0,len(feats)):
            r,s=feats[i]
            if(r>0):
                finale.append(s)

        dataframe = num_X.iloc[:, finale]

    else: # classification 
        
        c,d = num_X.shape
        names = np.arange(d)
        mode=SelectKBest(mutual_info_classif, k='all')
        mode.fit_transform(num_X,y)

        feats=sorted(zip(map(lambda c: round(c, 4), mode.scores_), names), reverse=True)

        finale=[]
        for i in range(0,len(feats)):
            r,s=feats[i]
            if(r>0):
                finale.append(s)

        dataframe = num_X.iloc[:, finale]
        

    ############# feature_generation #########################
    Non_linear= pd.DataFrame()
    linear=pd.DataFrame()
    m,n=dataframe.shape
    thrs=0.7
    for i in range(n):
        for j in range(i+1,n):
            if (i!=j) and (distcorr(dataframe.iloc[:,i],dataframe.iloc[:,j])!=0):
                if (distcorr(dataframe.iloc[:,i],dataframe.iloc[:,j])>0) and (distcorr(dataframe.iloc[:,i],dataframe.iloc[:,j])<thrs):
                    non_lin_X, non_lin_y =dataframe.iloc[:,i].to_frame(), dataframe.iloc[:,j]
                    model = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='rbf', kernel_params=None)
                    model.fit(non_lin_X, non_lin_y)
                    first = model.predict(non_lin_X)
                    first_feat_gen = pd.Series(first)
                    second_feat_gen = (non_lin_y -(first))
                    Non_linear = Non_linear.append(first_feat_gen,ignore_index=True)
                    Non_linear = Non_linear.append(second_feat_gen,ignore_index=True)
                elif (distcorr(dataframe.iloc[:,i],dataframe.iloc[:,j])>=thrs) and (distcorr(dataframe.iloc[:,i],dataframe.iloc[:,j])<=1):
                    lin_X, lin_y =dataframe.iloc[:,i].to_frame(), dataframe.iloc[:,j]
                    model = Ridge(alpha=1.0)
                    model.fit(lin_X, lin_y)
                    first_feat = model.predict(lin_X)
                    lin_first_feat = pd.Series(first_feat)
                    second_feat = (lin_y -(first_feat))
                    linear = linear.append(lin_first_feat, ignore_index=True)
                    linear = linear.append(second_feat, ignore_index=True)

    nonlinear_genereted=Non_linear.T
    linear_generated = linear.T
    print("no. of features generated by nonlinear features :", nonlinear_genereted.shape[1])
    print("no. of features generated by linear features:", linear_generated.shape[1])

    Generated_feats = pd.concat([nonlinear_genereted, linear_generated], axis=1)
    print("Total no. of generated features :", Generated_feats.shape[1])

    #################### Feature_selection in 2 steps###################
    if Generated_feats.shape[1]>0 :
        
        
        ############ 1st step of selction ##############
        if method == "RFE_rf" :
            if (y_column_types == "int64" or y_column_types == "float64"): 
                model = RandomForestRegressor()
            else:
                model = RandomForestClassifier()
                
            rfe = RFECV(model, step=1, cv=5)
            rfe.fit(Generated_feats,y)
            print("Optimal number of features after 1st step : %d" % rfe.n_features_)
            selected_feats_order = np.argsort(rfe.grid_scores_)[::-1]
            Data_X = pd.DataFrame()
            for i in range(rfe.n_features_):
                col = Generated_feats.iloc[:,selected_feats_order[i]]
                Data_X=Data_X.append(col)
            
            one_featsel= Data_X.transpose()

        elif method == "RFE_Lasso":
            if (y_column_types == "int64" or y_column_types == "float"):
                model = LassoLarsCV()
            else:
                model = LogisticRegressionCV(penalty="l1", solver='liblinear')
            
            rfe = RFECV(model, step=1, cv=5)
            rfe.fit(Generated_feats,y)

            print("Optimal number of features after 1st step : %d" % rfe.n_features_)

            selected_feats_order = np.argsort(rfe.grid_scores_)[::-1]

            Data_X = pd.DataFrame()
            for i in range(rfe.n_features_):
                col = Generated_feats.iloc[:,selected_feats_order[i]]
                Data_X=Data_X.append(col)

            one_featsel= Data_X.transpose()
            
        elif method == "Lasso" :
             if (y_column_types == "int64" or y_column_types == "float64"):
                 model = LassoLarsCV(eps=1e-8)
             else:
                 model = LogisticRegressionCV(penalty="l1", solver='liblinear')
            
             sfm=SelectFromModel(model, threshold=0.7)
             sfm.fit(Generated_feats, y)
             n_features = sfm.transform(Generated_feats).shape[1]
             print("optimal no. of features after lst step:", n_features)
             one_featsel = pd.DataFrame(sfm.transform(Generated_feats))
             
        elif method == "Boruta":
            if (y_column_types == "int64" or y_column_types == "float"):
                model = XGBRegressor()
            else:
                model = XGBClassifier()
            
            boruta = BorutaPy(model, n_estimators='auto', verbose=2)
            boruta.fit(Generated_feats.values,y.values)
            sel_index = Generated_feats.columns[boruta.support_]
            one_featsel=Generated_feats.loc[:, sel_index ]
            
        elif method == "XGB" :
            
            if (y_column_types == "int64" or y_column_types == "float64"):
                model = XGBRegressor()
            else:
                model = XGBClassifier()
                
            sfm=SelectFromModel(model, threshold=0.7)
            sfm.fit(Generated_feats, y)
            n_features = sfm.transform(Generated_feats).shape[1]
            print("optimal no. of features after lst step:", n_features)
            one_featsel = pd.DataFrame(sfm.transform(Generated_feats))
            
        #elif method == "Autoencoder" :
        
    ################# 2nd step of slections ###########
        o,p = one_featsel.shape
        Names = np.arange(p)
            
        if (y_column_types == "int64" or y_column_types == "float"):
            model=SelectKBest(mutual_info_regression, k='all')
        else:
             model=SelectKBest(mutual_info_classif, k='all')
                
        model.fit_transform(one_featsel,y)
    
        new_feat=sorted(zip(map(lambda o: round(o, 4), model.scores_), Names), reverse=True)
    
        sel_finale=[]
        for i in range(0,len(new_feat)):
            s,t=new_feat[i]
            if(s>0):
                sel_finale.append(t)
    
        second_featsel = one_featsel.iloc[:, sel_finale]
            
   ################ concat the original features & selected new features
   
        transformed_X = pd.concat([X, second_featsel], axis=1)
            

            
    else:
            
        transformed_X = X
            
            

    return None, transformed_X
