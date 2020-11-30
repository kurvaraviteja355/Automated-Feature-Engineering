# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random



df = pd.read_csv("PopularKids.csv")
df.head()

X = df.drop(["Goals"], axis=1)
y = df["Goals"]

categorical = X.select_dtypes(include=[object])
numerical = X.select_dtypes(exclude=[object])

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(categorical)
Data_cat=pd.DataFrame(enc.transform(categorical).toarray())
X_data = pd.concat([numerical, Data_cat], axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)
y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_data, y, train_size=0.8, random_state=42)

def get_fitness(individual):
    #rg = RandomForestRegressor(random_state=42)
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
high = high_mutation_fittest[:-1]
for item in high_mutation_fittest[:-1]:
    if item[1] == max(high_mutation_fitness):
        top_performer = item
        break
print("Total features included: " + str(top_performer[0].count(1)))

included_columns = [column for (column, binary_value) in zip(X.columns, top_performer[0]) if binary_value]
excluded_columns = [column for (column, binary_value) in zip(X.columns, top_performer[0]) if not binary_value]