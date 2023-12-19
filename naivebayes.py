import pickle
import numpy as np
import pandas as pd
from helpers import grouped_barplot, gamma_pdf_plot
from scipy.stats import gamma

data = pickle.load(open('sfdata.p', 'rb'))
C_list=[]



X_train=data['X_train'] 
print('X_train shape:',X_train.shape)
X_test=data['X_test']
print('X_train shape:',X_test.shape)
y_train=data['y_train']
print('y_train shape:',y_train.shape)
y_test=data['y_test']
print('y_train shape:',y_test.shape)
columns=data['columns'] #string name of each column in X_train and X_test
columns_types=data['column_types']
d={columns[i]:columns_types[i] for i in range(len(columns))}
print('Dictionary for columns and column_types: \n',d)

d_features=[k for k,v in d.items() if v=='d'] #discrete features
cont_features=[k for k,v in d.items() if v=='c'] #continuous features
print('Discrete features: ', d_features)
print('Continuous features: ', cont_features)


def calculate_unique_values() -> dict:
    ''' Returns a dictionary that contains all the unique values 
        of each **discrete feature** in X_train.
        
        The "unique values" of an array contains a complete set of
        the elements that can occur in the array, with no repeats.
        
        For example, the "unique values" of the array [-3, 5, -3, 4, 4, 2]
        are [-3, 5, 4, 2].
        
        Returns:
            - dict, where each key is a column name, and its value
            is an array containing all the unique values of that column.
    '''
    ## TODO: Implement the function.
    return {d_features[i]:np.unique(X_train[:,columns.index(d_features[i])]) for i in range(len(d_features))}

dict_return=calculate_unique_values()

print(f'Max unique values:',d_features[2],len(dict_return[d_features[2]]))
print(f'Min unique values:',d_features[-1],len(dict_return[d_features[-1]]))



class NaiveBayes():
    
    def __init__(self, D = None, C = 2):
        ''' Initializes the parameters of the Naive Bayes model.
        
        Inputs:
            - C (int): number of classes
            - D (int): number of features
        '''
        self.C = C
        self.D = D
        
        # Gamma distribution shape parameter (constant for all features)
        self.shape = 1.
        self.alpha=2


        
        # TODO: Initialize model parameters
        self.rho = None
        self.sigmas = None
        self.thetas = None
        
        
    def fit_rho(self, X, y):
        ''' Updates self.rho using the MLE from data X and y.
        '''
        self.X=X
        self.y=y
        self.rho= float(sum(np.array((y==1)))/len(y))
        pass
    
    def fit_sigmas(self, X, y):
        ''' Updates self.sigmas using the MLE from data X and y.
        '''
        self.X=X
        self.y=y 
        
        ###################### array sigma implementation ##################


        
        ################## Dictionary sigma implementation ###################
        cont_index=[0,1,2,3,5,7,8]
        test_sigma0={cont_features[i]:sum(self.X[self.y==0][:,cont_index[i]])/len((self.X[self.y==0][:,cont_index])) for i in range(len(cont_features))}
                                                        
        test_sigma1={cont_features[i]:sum(self.X[self.y==1][:,cont_index[i]])/len((self.X[self.y==1][:,cont_index])) for i in range(len(cont_features))}
        self.sigmas=[test_sigma0,test_sigma1]
        pass
    
    def fit_thetas(self, X, y):
        ''' Updates self.thetas using the MAP from data X and y.
        '''        
        
        ################ For 182 theta parameters ################
        disc_index=[4,6,9,10,11,12]
        disc_ind_dict={d_features[i]:disc_index[i] for i in range(len(d_features))}
        theta=np.array([])
        self.alpha=2
        self.X=X
        self.y=y
        C_list_sum=[]
        C1=0
        theta_big=[]
        for k in [0,1]:
            for feature in dict_return:  
                C_list_small=[]
                theta_Num=[]
                for value in dict_return[feature]:
                    C_list_small.append(sum(self.X[self.y==k][:,disc_ind_dict[feature]]==value))
                    theta_Num.append((2+sum(self.X[self.y==k][:,disc_ind_dict[feature]]==value)-1))
#                     print("theta Num",theta_Num,value)    

#                 print(C_list_small,y,feature,value)
                C1=sum(C_list_small)+len(dict_return[feature])
#                 print(C1)
                theta=theta_Num/C1
#                 print("****theta***",theta)
                theta_big.append(theta)
                C_list_sum.append(C1)
       

        # print(theta_big)
        Global_theta_list=[]
        for i in theta_big:
            Global_theta_list.extend(i)
            
        self.thetas=np.array(Global_theta_list,dtype=float) ###182 parameters


        
        
        
        
        pass
    
    def fit(self, X, y):
        ''' Fits the parameters of the Naive Bayes model using data X and y.
                
        Inputs:
        
            - X: np.ndarray of shape (N, D) containing D features and N examples
            - y: np.ndarray of shape (N, ) containing class labels for N examples
        '''
        
        ## DO NOT CHANGE THIS FUNCTION
        ## This function is called by the autograder.
        self.fit_rho(X, y)
        self.fit_sigmas(X, y)
        self.fit_thetas(X, y)
    
    def predict(self, X):
        ''' Returns an (N, ) array containing the predicted class labels for 
            the N examples in X: (N, D).
        '''
        ## This function is called by the autograder.
        
        y_pred=[]
        for x in X:
            y_pred_i=np.log(self.rho)+np.log(self.thetas)+gamma.logpdf(x,1,self.sigmas[1:]) 
            y_pred.append(y_pred_i)
        y_pred=np.array(y_pred)
          
        print(y_pred)
        return np.argmax(y_pred,axis=1).flatten()


        pass
    
    def generate_x(self, y: int):
        ''' Returns a (1, D) array containing a generated sample that belongs
            to class y.
        '''
        cont_var_0=[]
        disc_var_0=[]
        ## Sample continuous feature using gamma distribution
        for i in range(len(cont_features)):
            cont_var_0.append(np.random.gamma(1,self.sigmas[y][cont_features[i]]))
        
        ## Sample discrete feature using categorical distirbution. Shown here for 
        # the bed_type column which has five unique values
        disc_var_0=np.random.multinomial(1,self.thetas[0:5])
        
        
        return [cont_var_0,disc_var_0]
  
        
        pass
    
        
    ### Gradescope Test Case Helpers ###
    
    def test_rho(self) -> float:
        ''' Return the prior probability p(y = 1).
        '''
        return sum(np.array((self.y==1)))/len(self.y)

    def test_sigma1(self) -> float:
        ''' Return the learned scale parameter for the feature "bathrooms" for class y = 1.
        '''
#         return -1.
        return sum(self.X[self.y==1][:,1])/len(self.X[self.y==1][:,1])



    def test_sigma2(self) -> float:
        ''' Return the learned scale parameter for the feature "beds" for class y = 0.
        '''
        return sum(self.X[self.y==0][:,3])/len(self.X[self.y==0][:,3])

    def test_theta1(self) -> float:
        ''' Return the learned theta parameter for the feature "neighbourhood_cleansed",
            value "Mission", for class y = 0.
        '''
        self.alpha=2
        C_list=[]
        fea_neighbour=dict_return['neighbourhood_cleansed']
        for i in range(len(fea_neighbour)):
            C_list.append(sum(self.X[self.y==0][:,9]==fea_neighbour[i]))

        C_listplusone=np.array(C_list)+1
        return (self.alpha+sum(self.X[self.y==0][:,9]=='Mission') - 1)/sum(C_listplusone)

    def test_theta2(self) -> float:
        ''' Return the learned theta parameter for the feature "neighbourhood_cleansed",
            value "Western Addition", for class y = 1.
        '''
        self.alpha=2
        C_list=[]
        fea_neighbour=dict_return['neighbourhood_cleansed']
        for i in range(len(fea_neighbour)):
            C_list.append(sum(self.X[self.y==1][:,9]==fea_neighbour[i]))

        C_listplusone=np.array(C_list)+1
        return (self.alpha+sum(self.X[self.y==1][:,9]=='Western Addition') - 1)/sum(C_listplusone)
    
def main():
    model = NaiveBayes()
    model.fit(data['X_train'], data['y_train'])
    y_pred=model.predict(data['X_train'], data['y_train'])
    
    x_y0=model.generate_x(0) #continuous
    x_y1=model.generate_x(1)
    print("Generate new continous features with y=0: ",x_y0)
    print("Generate new continous features with y=1: ",x_y1)


    

    def compute_accuracy(y_true, y_hat):
        ''' Calculates the average accuracy (between [0, 1]) of predicted labels in y_hat (N, )
            given true labels y_true: (N, ).
        '''
        ## TODO: Implement the function.
        count=0
        for i in range(len(y_hat)):
            if y_true[i] == y_hat[i]:
                count+=1
        avg_acc = count/float(len(y_true))
                
        return 0.

if __name__ == "__main__":
    main()