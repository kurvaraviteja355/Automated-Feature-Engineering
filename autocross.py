# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from Task import DATATYPE_NUMBER, DATATYPE_CATEGORY_INT, DATATYPE_CATEGORY_STRING
import logging
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random





def create(X, X_column_types, y, y_column_types, arm, **kwargs):
  
    method = kwargs.get("method", None)
    #method = kwargs.get("method", "Binary_operators")
    #method = kwargs.get("method", "Binning")
    #method = kwargs.pop("method", "Cluster")
    categorical_cols = [c for c, t in zip(X.columns, X_column_types) if t in [DATATYPE_CATEGORY_INT, DATATYPE_CATEGORY_STRING]]
    numerical_cols = [c for c, t in zip(X.columns, X_column_types) if t == DATATYPE_NUMBER]
    categorical = X[categorical_cols]
    numerical = X[numerical_cols]
    # feature selection using Genetic Algorithm
    if method == "fs_GA":
        print("fs_GA")
        enc = OneHotEncoder()
        enc.fit(categorical)
        Data_cat=pd.DataFrame(enc.transform(categorical).toarray())
        X_data = pd.concat([numerical, Data_cat], axis=1)
        
        if y_column_types[0] == DATATYPE_NUMBER:
            y = y
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.fit(y)
            y = le.transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X_data, y, train_size=0.8, random_state=42)

        def get_fitness(individual):
            if y_column_types[0] == DATATYPE_NUMBER:
                rg = RandomForestRegressor(random_state=42)
            else:
                rg = RandomForestClassifier(random_state=42)
                
            columns = [column for (column, binary_value) in zip(X_train.columns, individual) if binary_value]
            training_set = X_train[columns]
            test_set = X_test[columns]
            rg.fit(training_set.values, y_train)
            preds = rg.predict(test_set.values)
            return 100 / np.sqrt(mean_squared_error(y_test, preds))

        individual = [1] * 100
        get_fitness(individual)

        def get_population_fitness(population):
            return sorted([(individual, get_fitness(individual)) for individual in population], key=lambda tup: tup[1], reverse=True)

        def crossover(individual_a, individual_b):
            crossing_point = random.randint(0, 99)
            offspring_a = individual_a[0:crossing_point] + individual_b[crossing_point:100]
            offspring_b = individual_b[0:crossing_point] + individual_a[crossing_point:100]
            return offspring_a, offspring_b

        def tournament(current_population):
            index = sorted(random.sample(range(0, 20), 5))
            tournament_members  = [current_population[i] for i in index]
            total_fitness = sum([individual[1] for individual in tournament_members])
            probabilities = [individual[1] / total_fitness for individual in tournament_members]
            index_a, index_b = np.random.choice(5, size=2, p=probabilities)
            return crossover(tournament_members[index_a][0], tournament_members[index_b][0])

        def mutation(individual):
            mutation_point = random.randint(0, 99)
            if(individual[mutation_point]):
                individual[mutation_point] = 0
            else:
                individual[mutation_point] = 1

        def build_next_generation(current_population, mutation_rate):
            next_generation = []
            next_generation.append(current_population[0][0]) # elitism
            next_generation.append(current_population[random.randint(1,19)][0]) # randomness
    
            for i in range(9): # tournaments
                offspring_a, offspring_b = tournament(current_population)
                next_generation.append(offspring_a)
                next_generation.append(offspring_b)
    
            for individual in next_generation: # mutation
                if(random.randint(1,mutation_rate) == 1):
                    mutation(individual)
            return next_generation
    

        def run_ga(current_population, num_of_generations, mutation_rate=1000):
            fittest_individuals = []
            for i in range(num_of_generations):
                current_population = get_population_fitness(current_population) # get pop fitness
                fittest_individuals.append(current_population[0]) # record fittest individual (for graphing and analysis)
                current_population = build_next_generation(current_population, mutation_rate) # make new population
                return fittest_individuals


        initial_population = [[random.randint(0, 1) for i in range(100)] for i in range(20)]
        high_mutation_fittest = run_ga(initial_population, 100, mutation_rate=5)



        high_mutation_fitness = [ind[1] for ind in high_mutation_fittest]
        for item in high_mutation_fittest[:-1]:
            if item[1] == max(high_mutation_fitness):
                top_performer = item
                break
        #print("Total features included: " + str(top_performer[0].count(1)))

        selected_features = [column for (column, binary_value) in zip(X.columns, top_performer[0]) if binary_value]
        excluded_features = [column for (column, binary_value) in zip(X.columns, top_performer[0]) if not binary_value]
        X = X[selected_features]
        categorical_cols = [c for c, t in zip(X.columns, X_column_types) if t in [DATATYPE_CATEGORY_INT, DATATYPE_CATEGORY_STRING]]
        numerical_cols = [c for c, t in zip(X.columns, X_column_types) if t == DATATYPE_NUMBER]
        categorical = X[categorical_cols]
        numerical = X[numerical_cols]
        
    if method == "Binary_operators":
        print("binaryoperators")
        Binary_operator = pd.DataFrame()
        #Apply binary operators
        for i in range(numerical.shape[1]):
            a = numerical.iloc[:,i]
            for j in range(i+1, numerical.shape[1]):
                b = numerical.iloc[:,j]
                result = a*b
                Binary_operator = pd.concat([Binary_operator, result], axis=1)
        # apply addition operation 
        for i in range(numerical.shape[1]):
            a = numerical.iloc[:,i]
            for j in range(i+1, numerical.shape[1]):
                b = numerical.iloc[:,j]
                result = a+b
                Binary_operator = pd.concat([Binary_operator, result], axis=1)
        numerical = Binary_operator.copy()
        
    if method == "Binning":
        print("Binning")
        num_discretizer=pd.DataFrame()
        for i in range(numerical.shape[1]):
            
            d_f1 = pd.DataFrame(pd.cut(numerical.iloc[:,i], 6, labels=False, duplicates='drop'))
            d_f2 = pd.DataFrame(pd.cut(numerical.iloc[:,i], 4, labels=False, duplicates='drop'))
            
            num_discretizer = pd.concat([num_discretizer, d_f1, d_f2], axis=1)
            
    else:
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
    # We cluster the categorical features by most frequently repated and kmeans clustering
    if method == "cluster":
        print("clustering")
        categorical_strg = [c for c, t in zip(categorical.columns, X_column_types) if t in [DATATYPE_CATEGORY_STRING]]
        categorical_int = [c for c, t in zip(categorical.columns, X_column_types) if t in [DATATYPE_CATEGORY_INT]]
        strg_cate = categorical[categorical_strg]
        int_cate = categorical[categorical_int]
        frequent = pd.DataFrame()
        frequent_col = []
        if len(strg_cate.columns)>=1:
            # clustering the string categorical variables by top 10 frequently repeated values
            for i in range(len(strg_cate.columns)):
                if (strg_cate[strg_cate.columns[i]].nunique() > 10):
                    frequent_col.append(strg_cate.columns[i])
                    n=10
                    top=strg_cate[strg_cate.columns[i]].value_counts()[:n].index.tolist()
                    for label in top:
                        strg_cate[label]= np.where(strg_cate[strg_cate.columns[i]]==label, 1, 0)
                        df1 = strg_cate[[strg_cate.columns[i]]+top]
                        frequent = pd.concat([frequent, df1.drop([strg_cate.columns[i]], axis=1)], axis=1)
            if len(frequent_col)>=1:
                strg_cate=strg_cate.drop(frequent_col, axis=1)
            else:
                strg_cate = strg_cate.copy()
                
        if len(int_cate.columns)>=1:
            # clustering the interger categorical variables by using kmeans clustering    
            int_col=[]    
            for i in range(len(int_cate.columns)):
                if (int_cate[int_cate.columns[i]].nunique() > 10):
                    x = int_cate.iloc[:,i:i+1]
                    kmeans = KMeans(10)
                    kmeans.fit(x)
                    cluster = kmeans.fit_predict(x)
                    int_cate[int_cate.columns[i] + '_cluster']=cluster
                    int_col.append(int_cate.columns[i])
            if len(int_col)>=1:
                int_cate = int_cate.drop(int_col, axis=1)
            else:
                int_cate = int_cate.copy()
            
        if (len(strg_cate.columns)>0 or len(int_cate.columns)>0):
            categorical = pd.concat([strg_cate, int_cate], axis=1)

        enc = OneHotEncoder()
        enc.fit(categorical)
        Data_cat=pd.DataFrame(enc.transform(categorical).toarray())
        
        if len(frequent_col)>=1:
            original_feats = pd.concat([numerical, Data_cat, frequent], axis=1)
            num_discret = pd.concat([numerical, Data_cat, frequent, num_discretizer], axis=1)
            
        else:
            original_feats = pd.concat([numerical, Data_cat], axis=1)
            num_discret = pd.concat([numerical, Data_cat, num_discretizer], axis=1)
            
            
            
        
    else:
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
        
