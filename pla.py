# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 23:49:28 2021

@author: Fernando Sanchez
@UNI: fs2664
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from matplotlib import cm
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import csv
def visualize_scatter(df, feat1=0, feat2=1, labels=2, weights=[-1, -1, 1],
                      title=''):
    """
        Scatter plot feat1 vs feat2.
        Assumes +/- binary labels.
        Plots first and second columns by default.
        Args:
          - df: dataframe with feat1, feat2, and labels
          - feat1: column name of first feature
          - feat2: column name of second feature
          - labels: column name of labels
          - weights: [w1, w2, b] 
    """

    # Draw color-coded scatter plot
    colors = pd.Series(['r' if label > 0 else 'b' for label in df[labels]])
    ax = df.plot(x=feat1, y=feat2, kind='scatter', c=colors)

    # Get scatter plot boundaries to define line boundaries
    xmin, xmax = ax.get_xlim()

    # Compute and draw line. ax + by + c = 0  =>  y = -a/b*x - c/b
    a = weights[0]
    b = weights[1]
    c = weights[2]

    def y(x):
        return (-a/b)*x - c/b

    line_start = (xmin, xmax)
    line_end = (y(xmin), y(xmax))
    line = mlines.Line2D(line_start, line_end, color='red')
    ax.add_line(line)


    if title == '':
        title = 'Scatter of feature %s vs %s' %(str(feat1), str(feat2))
    ax.set_title(title)

    plt.show()

def format_information(data_file):
    file = open(data_file);
    holder = np.loadtxt(file, delimiter=",")
    return holder;

def prediction(bias, weights, row):
    data_points = [row[a] for a in range(len(row)-1)];
    result = np.dot(weights, data_points);
    intercept_adder = 1 * bias;
    summation = result + intercept_adder;
    return summation;

def writeOutput(row, filename):#Will write the row of values to the output
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(row)
        
    return;


if __name__ == '__main__':
    initial_weights = [0, 0]; #How to accomodate the w0 or bias
    w0_bias = 0;
    container = format_information(sys.argv[1]);
    destination = sys.argv[2];
    holder = np.array(container);
    convergence = False;
    threshold = len(holder);
    correct = 0;
    while convergence == False:
        for row in holder:
            result = prediction(w0_bias, initial_weights, row);
            if row[-1] * result <= 0:
                copier = []
                for i in range(len(initial_weights)):
                    initial_weights[i] = initial_weights[i] + row[-1] * row[i];
                    copier.append(initial_weights[i])
                w0_bias = w0_bias + row[-1] * 1;
                
                
                copier.append(w0_bias);
                writeOutput(copier, destination);
                correct = 0;
                
            else:
                correct += 1
                if correct > threshold:
                    convergence = True;
            
                    
                        
        
                
    
            