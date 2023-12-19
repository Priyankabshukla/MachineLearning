#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

from operator import itemgetter


class Model(object):
    """
     Ridge Regression.
    """

    def fit(self, X, y, alpha=0):
        """
        Fits the ridge regression model to the training data.

        Arguments
        ----------
        X: nxp matrix of n examples with p independent variables
        y: response variable vector for n examples
        alpha: regularization parameter.
        """
        Iden= np.identity(len(X.T))  
        Iden[0,0] = 0
       	self.w = np.linalg.solve((X.T.dot(X)+alpha*Iden),X.T.dot(y))
        
    def predict(self, X):
        """
        Predicts the dependent variable of new data using the model.

        Arguments
        ----------
        X: nxp matrix of n examples with p covariates

        Returns
        ----------
        response variable vector for n examples
        """
        Yp = np.dot(X,self.w)
        return Yp

    def rmse(self, X, y):
        """
        Returns the RMSE(Root Mean Squared Error) when the model is validated.
            
        Arguments
        ----------
        X: nxp matrix of n examples with p covariates
        y: response variable vector for n examples
            
        Returns
        ----------
        RMSE when model is used to predict y
        """
        Yp = self.predict(X)
        rmse = np.sqrt(np.sum((y - Yp)**2)/len(Yp))
        return rmse
        


#run command:
#python ridge.py --X_train_set=data/Xtraining.csv --y_train_set=data/Ytraining.csv --X_val_set=data/Xvalidation.csv --y_val_set=data/Yvalidation.csv --y_test_set=data/Ytesting.csv --X_test_set=data/Xtesting.csv

if __name__ == '__main__':

    #Read command line arguments
    parser = argparse.ArgumentParser(description='Fit a Ridge Regression Model')
    parser.add_argument('--X_train_set', required=True, help='The file which contains the covariates of the training dataset.')
    parser.add_argument('--y_train_set', required=True, help='The file which contains the response of the training dataset.')
    parser.add_argument('--X_val_set', required=True, help='The file which contains the covariates of the validation dataset.')
    parser.add_argument('--y_val_set', required=True, help='The file which contains the response of the validation dataset.')
    parser.add_argument('--X_test_set', required=True, help='The file which containts the covariates of the testing dataset.')
    parser.add_argument('--y_test_set', required=True, help='The file which containts the response of the testing dataset.')
                        
    args = parser.parse_args()

    #Parse training dataset
    X_train = np.genfromtxt(args.X_train_set, delimiter=',')
    y_train = np.genfromtxt(args.y_train_set,delimiter=',')
    
    #Parse validation set
    X_val = np.genfromtxt(args.X_val_set, delimiter=',')
    y_val = np.genfromtxt(args.y_val_set, delimiter=',')
    
    #Parse testing set
    X_test = np.genfromtxt(args.X_test_set, delimiter=',')
    y_test = np.genfromtxt(args.y_test_set, delimiter=',')
    
    #find the best regularization parameter
    one = np.ones(len(X_train))
    X_train = np.vstack([one,X_train.T]).T  ## Stacking ones in first column
    X_val = np.vstack([np.ones(len(X_val)),X_val.T]).T  ## Stacking ones in first column
    X_test = np.vstack([np.ones(len(X_test)),X_test.T]).T
    A = [1,2,3,4,5,6,7,8,9]
    B = [-5,-4,-3,-2,-1,0]
    all_lam = []
    W = []
    
    for b in B:
        for a in A:
            lam = a*10**b
            all_lam.append(lam)
            mod = Model()
            mod.fit(X_train,y_train,lam)
            W.append(mod.w[0:11])
    k = np.arange(1,11)
    W = np.array(W)
    for i in k:
        plt.semilogx(all_lam,W.T[i],'-',label = f'k={i}')
        
    plt.legend()
    plt.xlabel(r'log $\lambda$')
    plt.ylabel(r'learned coefficient $\beta$')
    plt.show()
    plt.clf()
    

    #plot rmse versus lambda
    all_lam = []
    Rmse = []
    
    A = np.logspace(-5,-1)
    for lam in A:
        all_lam.append(lam)
        mod = Model()
        mod.fit(X_train,y_train,lam)
        Rmse.append(mod.rmse(X_val,y_val))
                
    
    plt.semilogx(all_lam,Rmse,'r')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'RMSE')
    plt.show()
    plt.clf()

    #plot predicted versus real value
    mod = Model()
    lambda_opt = 1e-3
    print('The optimum value of regularization parameter: ', lambda_opt)
    mod.fit(X_train,y_train,lambda_opt)
    yp = mod.predict(X_test)
    plt.scatter(y_test,yp)
    plt.show()
    plt.clf()
    
