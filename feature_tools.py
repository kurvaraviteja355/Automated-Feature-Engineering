# -*- coding: utf-8 -*-
def create(X, X_column_types, y, y_column_types, arm, **kwargs):
    import numpy as np
    import pandas as pd
    import featuretools as ft
    from featuretools import variable_types as vtypes
    from sklearn.feature_selection import RFECV
    from sklearn.ensemble import RandomForestRegressor
    
    
    global org_keys
    org_keys = X.keys()
    

    X['Target']=y
    Target = X.keys()[-1]
    
    #correlated featyures
    corr = X.corr()
    corr_features = corr[Target].sort_values(ascending=False)
    
    # checking whether the data is having categorical features
    categorical_feats = X.dtypes[X.dtypes == "object"].index
    print("Number of Categorical features: ", len(categorical_feats))
    cat_feats = len(categorical_feats)
    
    cat_variables = X.select_dtypes(include=[object]) 
    date_time = X.select_dtypes(include=["datetime64"])
    
    boolean_variables = X.select_dtypes(include=[bool])
    
    #create the ensityset
    es = ft.EntitySet("feature_data")
    
    if cat_feats > 0:
        
        for col in cat_variables.columns:
            try:
                cat_variables[col]=pd.to_datetime(cat_variables[col])
            except:
                continue
        
        object_time = cat_variables.select_dtypes(include=["datetime64"])
        
        if object_time.shape[1]>0:
            
            cat_variables = cat_variables.drop(object_time, axis=1)
        else:
            cat_variables = cat_variables.copy()
            
        date_time = pd.concat([date_time, object_time], axis=1)
        
    if date_time.shape[1]>0:   
        time=[]
        for i in range (date_time.shape[1]):
            l = date_time.keys()[i]
            time.append(l)
        
    if (cat_feats > 0 or boolean_variables.shape[1] > 0):
        
        data_variable_types={}
        for col in cat_variables.columns:
            data_variable_types[col] = vtypes.Categorical
        for col in boolean_variables.columns:
            data_variable_types[col]=vtypes.Boolean
        
        
    if (cat_feats > 0 and date_time.shape[1] > 0):
        
        es = es.entity_from_dataframe(entity_id="data", dataframe=X, time_index = time[0],
                                      variable_types= data_variable_types, index="Id")
    if (cat_feats > 0 and date_time.shape[1] == 0):
             es = es.entity_from_dataframe(entity_id="data", dataframe=X, varaible_types= data_variable_types, 
                                           index="Id")
    if (date_time.shape[1] > 0 and cat_variables.shape[1] == 0):
        es = es.entity_from_dataframe(entity_id="data", dataframe=X, time_index= time[0], index="Id")
    
    else:
        es = es.entity_from_dataframe(entity_id="data", dataframe=X, index="Id")

    
    # create the normalize entities with correlated features
    if (date_time.shape[1] > 0 and len(time)>1):
        es = es.normalize_entity(base_entity_id="data", new_entity_id= corr_features.keys()[1],
                                 index=corr_features.keys()[1], additional_variables= time[1:])
    else:
        es = es.normalize_entity(base_entity_id="data", new_entity_id= corr_features.keys()[1],
                                 index=corr_features.keys()[1])
        
    es = es.normalize_entity(base_entity_id="data", new_entity_id= corr_features.keys()[2],
                             index=corr_features.keys()[2])   
   

    # Adding the relationship now

    train_matrix, features = ft.dfs(entityset=es, target_entity="data",
                                    max_depth=2, verbose=True)
    
    #if train_matrix

    # fill the missing  values
    train_matrix.fillna(train_matrix.median(), inplace=True)

    Xtrain = train_matrix.drop(['Target'], axis=1)
    target = train_matrix['Target']

    
    # Seperate the generated_feats and Original_feats
    if date_time.shape[1]>0:
        col_rm = set(org_keys)-set(time)
        transformed_X = Xtrain.drop(list(col_rm), axis=1)
    else:
        transformed_X = Xtrain.drop(org_keys, axis=1)
        
   
    return None, transformed_X


