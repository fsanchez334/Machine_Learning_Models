# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:08:48 2021

@author: ferna
"""
import sys
import numpy as np
import csv
import pandas as pd

def format_information(data_file):
    file = open(data_file);
    holder = pd.read_csv(file, header = None)
    return holder;

def normalize_data(dataframe, age_mean, weight_mean, age_std, weight_std):
    numper = dataframe.to_numpy();
    shaper = numper.transpose();
    
    for row in range(0,2,1):
        for i in range(len(shaper[row])-1):
            if row == 0: #Indicates we are in the age column
                normalized = (shaper[row, i] - age_mean) / age_std;
                shaper[row, i] = normalized
            else:
                normalized= (shaper[row, i] - weight_mean) / weight_std
                shaper[row, i] = normalized;
    return shaper; #Data has been normalized

def reshape_intercept(normalized_data):
    information_normal = normalized_data.transpose();#Information is reset in numpy format - now that it has been normalized
    intercept_column = [1 for x in range(len(normalized_data[0]))] #Making the column for the intercerpt
    updated = np.insert(information_normal, 0, intercept_column, axis=1);
    return updated;

def linear_prediction(data_numpy, betas, n):
    transfer_betas = betas; #Will allow for us to manipulate the betas
    errors = [];
    for row in data_numpy:
        features_subset = row[0:len(row)-1];
        dot_product = np.dot(transfer_betas, features_subset);
        result = dot_product - row[-1];
        error_squared = round((result ** 2), 4);
        errors.append(error_squared);
    
    marker = (1 / (2 * n))
    result = round(marker * sum(errors), 4);
    return result;

    #This function just calculates the error function that comes with the betas -> in theory, it should give me
    #decreasing values as the betas are changing

def beta_change(data_numpy, betas, n, learning_rate):
    old_betas = betas;
    individual_products = []
    new_betas = [];
    for index in range(len(old_betas)):
        for row in data_numpy:
            features_subset = row[0:len(row)-1];
            dot_product = np.dot(old_betas, features_subset);
            result = dot_product - row[-1];
            product = result * features_subset[index]; #Multiplies difference between prediction and label with feature value
            individual_products.append(product);
        
        sum_product = sum(individual_products);
        greater_difference = learning_rate * (1 / n) * sum_product;
        improved_beta = old_betas[index] - greater_difference;
        new_betas.append(improved_beta);
    
    return new_betas;

def writeOutput(final_information, filename):#Will write the row of values to the output
    for row in final_information:
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(row)
        
    return;

    
    
if __name__ == '__main__':
    information = format_information(sys.argv[1]);
    destination = sys.argv[2]; 
    
    #Here, first calculate the means and standard deviation for each feature
    age_mean = (information.iloc[:, 0]).mean(axis=0);
    weight_mean = (information.iloc[:, 1]).mean(axis = 0);
    
    age_std  =(information.iloc[:, 0]).std();
    weight_std = (information.iloc[:, 1]).std();
    normal = normalize_data(information, age_mean, weight_mean, age_std, weight_std);
    betas = [0, 0, 0];
    n = len(normal[0]);
    updated = reshape_intercept(normal)#From here, we then go through the information to perform the dot product
    
    #Here is we begin testing for one of the learning rates
    l_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10];
    diction_rates = {
            0.001: [],
            0.005: [],
            0.01: [],
            0.05: [],
            0.1: [],
            0.5: [],
            1: [],
            5: [],
            10: []
    }
    error_rates = {
            0.001: [],
            0.005: [],
            0.01: [],
            0.05: [],
            0.1: [],
            0.5: [],
            1: [],
            5: [],
            10: []  
    }
    holder_final = []
    for index in range(len(l_rates)):
        for i in range(100):
            train_rate = l_rates[index]
            risk = linear_prediction(updated, betas, n);
            updated_b = beta_change(updated, betas, n, train_rate);
            betas = updated_b;
            diction_rates[l_rates[index]].append(risk);
            if i == 99:
                lister = [l_rates[index], 100];
                rounded = [round(num, 4) for num in betas]
                lister.extend(rounded);
                holder_final.append(lister)
    
    for keys in diction_rates:
        error_rates[keys].append(diction_rates[keys][-1]);
    
    
    #Holder final has the information that we need -  now we have to write the information to the output 
    writeOutput(holder_final, destination)
    tester_betas = [0,0,0];
    
    results = []
    sole = {
        0.067: []
   
    }
    for i in range(250):
            train_rate = 0.067
            risk = linear_prediction(updated, tester_betas, n);
            updated_b = beta_change(updated, tester_betas, n, train_rate);
            tester_betas = updated_b;
            sole[train_rate].append(risk);
            if i == 249:
                lister = [train_rate, 250];
                rounded = [round(num, 4) for num in tester_betas]
                lister.extend(rounded);
                results.append(lister)
        
    writeOutput(results, destination);
    
            
    
        
        
    