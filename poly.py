#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import pandas as pd 
from matplotlib import pyplot as plt

from operator import itemgetter


class Model(object):
    """
     Polynomial Regression.
    """

    def fit(self, X, y, k):
        """
        Fits the polynomial regression model to the training data.

        Arguments
        ----------
        X: nx1 matrix of n examples
        y: response variable vector for n examples
        k: polynomial degree
        """
        
        X_new = np.ones(len(X))[:,np.newaxis]
        for i in np.arange(1,k+1):
            X_new = np.hstack([X_new,X**int(i)])
        self.w = np.linalg.solve(X_new.T.dot(X_new),X_new.T.dot(y))
        self.x = X_new
        self.k = k
#         raise NotImplementedError

    def predict(self, X):
        """
        Predicts the dependent variable of new data using the model.

        Arguments
        ----------
        X: nx1 matrix of n examples

        Returns
        ----------
        response variable vector for n examples
        """
        X_new = np.ones(len(X))[:,np.newaxis]
        for i in np.arange(1,self.k+1):
            X_new = np.hstack([X_new,X**int(i)])
        Yp = np.dot(X_new,self.w)
        return Yp
#         raise NotImplementedError

    def rmse(self, X, y):
        """
        Returns the RMSE(Root Mean Squared Error) when the model is validated.
        
        Arguments
        ----------
        X: nx1 matrix of n examples
        y: response variable vector for n examples
        
        Returns
        ----------
        RMSE when model is used to predict y
        """
        Yp = self.predict(X)
        rmse = np.sqrt((np.sum((Yp - y)**2))/len(y))
        return rmse
#         raise NotImplementedError


#run command:
#python poly.py --data=data/poly_reg_data.csv

if __name__ == '__main__':

    #Read command line arguments
    parser = argparse.ArgumentParser(description='Fit a Polynomial Regression Model')
    parser.add_argument('--data', required=True, help='The file which contains the dataset.')
                        
    args = parser.parse_args()

    input_data = pd.read_csv(args.data)
    
    n = len(input_data['y'])
    n_train = 25
    n_val = n - n_train

    x = input_data['x']
    x_train = x[:n_train][:,None]
    x_val = x[n_train:][:,None]

    y= input_data['y']
    y_train = y[:n_train][:,None]
    y_val = y[n_train:][:,None]
    
    Rmse_train = []
    Rmse_val = []
    K = np.linspace(1,10,10)
    Xval_new = np.ones(n_val)[:,np.newaxis]
    for k in K:
        mod = Model()
        mod.fit(x_train,y_train,k)
        rmse = mod.rmse(x_train,y_train)
        Rmse_train.append(rmse)
        rmse1 = mod.rmse(x_val,y_val)
        Rmse_val.append(rmse1)

    #plot validation rmse versus k
    plt.plot(K,Rmse_val,'r-')
    plt.xlabel('polynomial degree (k)')
    plt.ylabel('Rmse_Val')
    plt.show()
    plt.clf()

    #plot training rmse versus k
    plt.plot(K,Rmse_train,'b-')
    plt.xlabel('polynomial degree (k)')
    plt.ylabel('Rmse_Training')
    plt.show()
    plt.clf()
    #plot fitted polynomial curve versus k as well as the scattered training data points 
    k_prime = [1,3,5,10]
    X_seq = np.linspace(min(x_train), max(x_train),200).reshape(-1,1)
    for i in k_prime:
        mod = Model()
        mod.fit(x_train,y_train,i)
        plt.scatter(x_train,y_train)
        Y_seq = mod.predict(X_seq)
        plt.plot(X_seq, Y_seq,"red",label=f"k={i}")
        plt.xlabel("x-data")
        plt.ylabel("y-predicted")
        plt.legend()
        plt.show()
        plt.clf()
    

