# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from Task import DATATYPE_NUMBER, DATATYPE_CATEGORY_INT, DATATYPE_CATEGORY_STRING
import logging





def create(X, X_column_types, y, y_column_types, arm, **kwargs):
    
    categorical_cols = [c for c, t in zip(X.columns, X_column_types) if t in [DATATYPE_CATEGORY_INT, DATATYPE_CATEGORY_STRING]]
    numerical_cols = [c for c, t in zip(X.columns, X_column_types) if t == DATATYPE_NUMBER]
    categorical = X[categorical_cols]
    numerical = X[numerical_cols]
    
    # discritize the numerical features
    num_discretizer=pd.DataFrame()
    for i in range(numerical.shape[1]):
        d_f = pd.DataFrame(pd.cut(numerical.iloc[:,i], 10, labels=False))
        d_f2 = pd.DataFrame(pd.cut(numerical.iloc[:,i], 5, labels=False))
        d_f3 = pd.DataFrame(pd.cut(numerical.iloc[:,i], 4, labels=False))
        d_f4 = pd.DataFrame(pd.cut(numerical.iloc[:,i], 3, labels=False))
    
        num_discretizer = pd.concat([num_discretizer, d_f, d_f2, d_f3, d_f4], axis=1)
    
    # function to rename the duplicate columns
    def df_column_uniquify(df):
        df_columns = df.columns
        new_columns = []
        for item in df_columns:
            counter = 0
            newitem = item
            while newitem in new_columns:
                counter += 1
                newitem = "{}_{}".format(item, counter)
            new_columns.append(newitem)
        df.columns = new_columns
        return df
    
    
    num_discretizer=df_column_uniquify(num_discretizer)
    
    
    # Categorical features encoding 
    cat_list=[]
    for i in range(categorical.shape[1]):
        if (len(categorical.iloc[:, i].unique()) >= 2):
            cat_list.append(categorical.keys()[i])
            
    categorical = categorical[cat_list]
    # One hot encode the categorical_features      
    #Data_cat = pd.get_dummies(categorical)
    enc = OneHotEncoder()
    enc.fit(categorical)
    Data_cat=pd.DataFrame(enc.transform(categorical).toarray())
    
    original_feats = pd.concat([numerical, Data_cat], axis=1)
    
    num_discret = pd.concat([numerical, Data_cat, num_discretizer], axis=1)

   
    
   #Select the best half of discretized features by Mini batch gradient descent 
   
    #clf = SGDClassifier(loss="log", penalty="l1")
    
    mini_batches = [] 
    batch_size=32
    data = np.hstack((num_discret, (y.values).reshape(-1,1)))
    #data =pd.concat([num_discretizer, y], axis=1) 
    np.random.shuffle(data) 
    n_minibatches = data.shape[0] // batch_size 
    i = 0
  
    for i in range(n_minibatches + 1): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    if data.shape[0] % batch_size != 0: 
        mini_batch = data[i * batch_size:data.shape[0]] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
        
    if (y_column_types[0] == DATATYPE_NUMBER):
       
        model = SGDRegressor(loss="squared_loss", penalty="l1")
        for X_mini, Y_mini in mini_batches:
            model.partial_fit(X_mini, Y_mini)
        coefs=model.coef_
    else:
        model = SGDClassifier(loss="log", penalty="l1")
        for X_mini, Y_mini in mini_batches:
            model.partial_fit(X_mini, Y_mini, classes=np.unique(y))
        coefs=model.coef_[0]
            
    num = len(numerical.columns)+len(Data_cat.columns)    
    #coefs=model.coef_
    h=np.argsort(coefs[num:])[::-1][:int(num_discretizer.shape[1]/2)]
    best_half_sorted = [x+num for x in h]
    best_dicretized = num_discret.iloc[:,best_half_sorted]
    
    
    total = pd.concat([categorical, best_dicretized], axis=1)
    
    # one hot encode the interger discretized features
    enc = OneHotEncoder()
    enc.fit(best_dicretized)
    dicretized_ohe=pd.DataFrame(enc.transform(best_dicretized).toarray())
   
    
    

    
    # combine cat_ohe and disretized_ohe  features 
    Data = pd.concat([Data_cat, dicretized_ohe], axis=1)
    
    # Rename the features which has duplicates 
    Data = df_column_uniquify(Data)
    

    second_order = pd.DataFrame()
    final_feats = pd.DataFrame()
    cnt = 0
    cnt_1 = 0
    for i in range(len(total.columns)-1):
        a= Data.iloc[:,[o for o in range(cnt, cnt+len(total.iloc[:, i].unique()))]]
        cnt = cnt+len(total.iloc[:, i].unique())
        cnt_1 = cnt
        for j in range(i+1, len(total.columns)):
            b= Data.iloc[:,[p for p in range(cnt_1, cnt_1+len(total.iloc[:, j].unique()))]]
            cnt_1 = cnt_1+len(total.iloc[:, j].unique())
            first = pd.DataFrame()
            for k in range(a.shape[0]):
                c = a.iloc[[k]].values
                d = b.iloc[[k]].values
        
                result = np.outer(c, d).ravel()
                first=first.append(pd.Series(result), ignore_index=True)
        
   
            second_order = pd.concat([second_order, first], axis =1)
    second_order = df_column_uniquify(second_order)
    
    firstorder_select = pd.concat([original_feats, second_order], axis=1)
        
    # slect the second order features using Logistic regression
    #clf = SGDClassifier(loss="log", penalty="l1")

    mini_batches = [] 
    batch_size=32
    data = np.hstack((firstorder_select, (y.values).reshape(-1,1))) 
    #data = pd.concat([second_order, y], axis=1)
    np.random.shuffle(data) 
    n_minibatches = data.shape[0] // batch_size 
    i = 0
  
    for i in range(n_minibatches + 1): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    if data.shape[0] % batch_size != 0: 
        mini_batch = data[i * batch_size:data.shape[0]] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
        #create_mini_batches(gen_feats, y, 32)
        
    if (y_column_types[0] == DATATYPE_NUMBER):
        model = SGDRegressor(loss="squared_loss", penalty="l1")
        for X_mini, Y_mini in mini_batches:
            model.partial_fit(X_mini, Y_mini)
        coefs=model.coef_
    else:
        model = SGDClassifier(loss="log", penalty="l1")
        for X_mini, Y_mini in mini_batches:
            model.partial_fit(X_mini, Y_mini, classes=np.unique(y))
        coefs=model.coef_[0]
            
    num1 = len(original_feats.columns)   
    #selected top 10 features
    g=np.argsort(coefs[num1:])[::-1][:10]
    selected_sorted=[x+num1 for x in g]
    selected_best = firstorder_select.iloc[:, selected_sorted]
    selected = selected_best.copy()
    new_col_types = X_column_types+[DATATYPE_CATEGORY_INT]*len(selected_best.columns)
    total_feats = pd.concat([original_feats, selected_best], axis=1)
    final_feats = pd.concat([X, selected_best], axis=1)            
    
    # higher order features generation
   
    if len(categorical.columns)>2:
        for i in range(len(categorical.columns)-2):
           cnt = 0
           Higher_order = pd.DataFrame()
           for i in range(len(total.columns)):     
               a= Data.iloc[:,[o for o in range(cnt, cnt+len(total.iloc[:, i].unique()))]]
               cnt = cnt+len(total.iloc[:, i].unique())
               for j in range(selected_best.shape[1]):
                   b= selected_best.iloc[:,j]
                   second = pd.DataFrame()
                   for k in range(a.shape[0]):
                       c = a.iloc[[k]].values
                       d = b.iloc[[k]].values
                       result_1 = np.outer(c, d).ravel()
                       second=second.append(pd.Series(result_1), ignore_index=True)
                       
                   Higher_order = pd.concat([Higher_order, second], axis =1)
                   
           Higher_order=df_column_uniquify(Higher_order)
           
           High_order_sel = pd.concat([total_feats, Higher_order], axis=1)
           mini_batches = [] 
           batch_size=32
           data = np.hstack((High_order_sel, (y.values).reshape(-1,1))) 
           #data = pd.concat([Higher_order, y], axis=1)
           np.random.shuffle(data) 
           n_minibatches = data.shape[0] // batch_size 
           i = 0
           for i in range(n_minibatches + 1):
               mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
               X_mini = mini_batch[:, :-1] 
               Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
               mini_batches.append((X_mini, Y_mini))
           if data.shape[0] % batch_size != 0:
               mini_batch = data[i * batch_size:data.shape[0]] 
               X_mini = mini_batch[:, :-1] 
               Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
               mini_batches.append((X_mini, Y_mini))
               #create_mini_batches(gen_feats, y, 32)
           if (y_column_types[0] == DATATYPE_NUMBER):
               model = SGDRegressor(loss="squared_loss", penalty="l1")
               for X_mini, Y_mini in mini_batches:
                   model.partial_fit(X_mini, Y_mini)
               coefs=model.coef_
           else:
               model = SGDClassifier(loss="log", penalty="l1")
               for X_mini, Y_mini in mini_batches:
                   model.partial_fit(X_mini, Y_mini, classes=np.unique(y))
               coefs=model.coef_[0]
               
               
           #coefs=model.coef_
           num2 = len(total_feats.columns)
           sort=np.argsort(coefs[num2:])[::-1][:5]
           selected_sorted=[x+num2 for x in sort]
           selected_best = High_order_sel.iloc[:, selected_sorted]
           selected = pd.concat([selected, selected_best], axis=1)
           total_feats = pd.concat([total_feats, selected_best], axis=1)
           final_feats = pd.concat([final_feats, selected_best], axis=1)
           
    
        transformed_X = final_feats
        new_col_types = X_column_types+[DATATYPE_CATEGORY_INT]*len(selected.columns)
        
           
    else:
        
        transformed_X = final_feats
        
    return None, transformed_X, new_col_types
        


