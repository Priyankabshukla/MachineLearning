import pickle
import numpy as np
import matplotlib.pyplot as plt

from helpers import plot_confusion_matrix

data = pickle.load( open( "data.p", "rb" ) )

def flatten_data(X):
    ''' Returns a flattened version of X.
    
    Inputs:
        - X (np.ndarray): 3D array of shape (N, S1, S2)
        
    Returns:
        - a 2D np.ndarray of shape (N, S1 * S2)
    '''
    ## This function is called by the autograder.
    return X.reshape(10000,28*28)
    pass
    
def unflatten_data(X_flat):
    ''' Returns an unflattened version of X.
    
    Inputs:
        - X_flat (np.ndarray): 2D array of shape (N, S1 x S2)
        
    Returns:
        - a 3D np.ndarray of shape(N, S1, S2)
    '''
    ## This function is called by the autograder.
    return X_flat.reshape(10000,28,28)
    pass

def forward_linear(M, v):
    ''' Returns the output of a linear layer with weights M, applied 
        to the 1D array v.
        
    Inputs:
        - M (np.ndarray): 2D array of shape (S1, S2)
        - v (np.ndarray): 1D array of shape (S2, )
        
    Returns:
        a 1D array of shape (S1,) that is the result of multiplying
        the weights in M, times the input in v.
    '''
    ## This function is called by the autograder.
    return np.dot(M,v)
    pass

def forward_tanh(v):
    ''' Returns the output of applying an element-wise tanh activation 
    function to the activations v.
        
    Inputs: 
        - v (np.ndarray): 1D array of shape (S1, )
        
    Returns: 
        a 1D array of shape (S1, ) that is the result of applying 
        the element-wise tanh function to v.
    '''
    ## This function is called by the autograder.
    return np.tanh(v)
    pass

def forward_softmax(v):
    ''' Returns the output of applying the softmax function to the
        vector v.
        
    Inputs:
        - v (np.ndarray): 1D aray of shape (S1, )
        
    Returns:
        - a 1D array of shape (S1, ) that is the result of applying the
        softmax function to the vector v
    '''
    ## This function is called by the autograder.
    return np.exp(v)/np.sum(np.exp(v),axis=0)
    pass

def cross_entropy(y_true, y_hat) -> float:
    ''' Returns the average cross-entropy loss of the predictions in y_hat
        given the true labels y_true.
    
    Inputs:
        - y_true (np.ndarray): 2D one-hot-encoded array of shape (N, C) 
        with N datapoints and C classes
        - y_hat (np.ndarray): 2D array of shape (N, C) containing predicted
        probabilities for each class
        
    Returns:
        - the average (over data-points) cross-entropy loss of the predictions
          given the true labels y_true
    '''
    return np.average(-np.sum(y_true*np.log(y_hat),axis=1))
    pass

def add_bias(v):
    ''' Appends a bias term to the beginning of the array v.
    
    Input:
        - v (np.ndarray): 1D array of shape (D,)
        
    Returns:
        - a 1D array of shape (D + 1, ) where the first
        column contains a 1
    '''
    return np.concatenate((np.array([1.]), v), axis = 0)


class NeuralNetwork():
    
    def __init__(self, d0: int, 
                 d1: int, 
                 d2: int,
                 lamb: float = 0.,
                 seed = 701):
        ''' Initializes a neural network with one hidden layer,
            using the dimensions specified in d0, d1, and d2.
        
        Inputs:
            - d0 (int): input dimension (excluding bias term)
            - d1 (int): hidden layer dimension (excluding bias term)
            - d2 (int): output dimension
            - lambda (float): regularization hyperparameter
        '''
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        
        self.lamb = lamb
        
        # Initialize network parameters between [-0.5, 0.5]
        # Do not change the below code, as it is called by the autograder!
        np.random.seed(seed)
        self.W = np.random.uniform(size = (self.d1, self.d0 + 1)) - 0.5
        self.V = np.random.uniform(size = (self.d2, self.d1 + 1)) - 0.5
        
    def dloss_dV(self, x, y, y_hat, z):
        ''' Returns the gradient of the cross-entropy loss on the datapoint
            (x, y) with predicted probabilities y_hat, with respect to self.V.
            
            Note that you may not need to use all of the input arguments.
            
        Inputs:
            x (np.ndarray): 1D array of shape (d0, ) containing features 
            y (np.ndarray): 1D array of shape (d2, ) containing true label
            y_hat (np.ndarray): 1D array of shape (d2, ) containing the model's predicted
                probabilities for x
            z (np.ndarray): 1D array of shape (d1, ) containing the activations from
                the model's forward pass on x
                
        Outputs:
            a 2D array of shape self.V.shape where each element corresponds to 
            the gradient of the cross-entropy loss, with respect to V.
        '''
        k=add_bias(z)
        dl_db=-y+y_hat
        grad=np.multiply(dl_db[:,np.newaxis],k[:,np.newaxis].T)+(2*self.lamb*self.V)
        assert grad.shape == self.V.shape
        return grad
    
    def dloss_dW(self, x, y, y_hat, z):
        ''' Returns the gradient of the cross-entropy loss on the datapoint
            (x, y) with predicted probabilities y_hat, with respect to self.W.
            
        Inputs:
            x (np.ndarray): 1D array of shape (d0, ) containing features 
            y (np.ndarray): 1D array of shape (d2, ) containing true label
            y_hat (np.ndarray): 1D array of shape (d2, ) containing the model's predicted
                probabilities for x
            z (np.ndarray): 1D array of shape (d1, ) containing the activations from
                the model's forward pass on x
                
        Outputs:
            a 2D array of shape self.W.shape where each element corresponds to 
            the gradient of the cross-entropy loss, with respect to W.
        '''
        
        k=add_bias(x)
        dl_db=-y+y_hat
        Vbar=self.V[:,1:]
        dl_dz=np.dot(Vbar.T,dl_db[:,np.newaxis])
        
        zprime=np.dot(self.W,k)


        dl_dzprime=dl_dz*(1-np.tanh(zprime[:,np.newaxis])**2)
        grad=np.dot(dl_dzprime,k[:,np.newaxis].T)+(2*self.lamb*self.W)

        assert grad.shape == self.W.shape
        return grad
        
    
    def forward(self, x):
        ''' Returns the forward pass (predicted probabilities) of the
            neural network on input x.
            
        Inputs:
            - x (np.ndarray): 1D array of shape (d0, ) containing features
            
        Outputs (in order):
            - zprime (np.ndarray): 1D array of shape (self.d1, ) containing
                hidden layer input before tanh activations
            - z (np.ndarray): 1D array of shape (self.d1, ) containing
                hidden layer output after tanh activations
            - b (np.ndarray): 1D array of shape (self.d2, ) containing
                output layer input before softmax activations
            - y_hat (np.ndarray): 1D array of shape (self.d2, ) containing
                output layer output after softmax activations
        '''
        assert x.shape[0] == self.d0
        x=add_bias(x)
        
        
        zprime = forward_linear(self.W, x)
        z = forward_tanh(zprime)
        k=add_bias(z)
        
        b = forward_linear(self.V,k)
        y_hat = forward_softmax(b)

        
        return zprime, z, b, y_hat
    def fit(self, X, y, eta, num_epochs = 50,  Xtest=None, y_test=None,test=False):
        ''' Fits the neural network parameters by running gradient 
            descent using data X and y.
            
        Inputs:
            - X (np.ndarray): 2D array of shape (N, d0) containing
              N datapoints
          -   y (np.ndarray): 1D array of shape (N, ) containing labels
              for each datapoint
        '''

        self.entropy=np.array([])
        self.entropy_test=np.array([])
        for epoch in trange(0,num_epochs):
            y_hat_big=[]

            for i,x in enumerate(X):

                zprime,z,b,y_hat=self.forward(x)
                y_hat_big += [y_hat]
                dl_dV=self.dloss_dV(x, y[i], y_hat, z)
                dl_dW=self.dloss_dW(x, y[i], y_hat, z)
                self.V=self.V-(eta*dl_dV)
                self.W=self.W-(eta*dl_dW)           

            self.entropy=np.append(self.entropy,cross_entropy(y, np.array(y_hat_big)))
            
            if test:
                y_pred_list=[]
                for x in Xtest:
                    zprime,z,b,y_pred=self.forward(x)
                    y_pred_list+=[y_pred]
                self.entropy_test=np.append(self.entropy_test,cross_entropy(y_test,np.array(y_pred_list)))
                                
        
        pass
    
    def predict_class(self, X):
        ''' Predicts a single most-likely class for each datapoint in X.
        
        Inputs:
            - X (np.ndarray): array of shape (N, self.d0)
            
        Outputs:
            a 1D array of integer class values corresponding to the most
            likely class for each example
        '''
        y_pred_list=np.array([])

        for x in X:
            zprime,z,b,y_pred=self.forward(x)
            
            y_pred_list=np.append(y_pred_list,np.argmax(y_pred))
            
 
        return y_pred_list
        pass
    
from sklearn.preprocessing import StandardScaler


def train_MNIST_model(eta, num_epochs):
    ''' Starter code for a function that trains a NeuralNetwork model
        on the MNIST data.
        
    Inputs:
        eta: learning rate used during training
        num_epochs: number of training epochs
    '''
    
    # Flatten the image features
    X_train_flat = flatten_data(data["X_train"])
    X_test_flat = flatten_data(data["X_test"])
    
    # Re-scale the features to have mean 0 and unit variance
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    ## TODO: Initialize and fit your neural network.
    model=NeuralNetwork(784,100,10,lamb=1e-4)
    model.fit(flatten_data(data["X_train"]),data['y_train'],eta=1e-4,num_epochs=50,Xtest=flatten_data(data["X_test"]),y_test=data["y_test"],test=True)
    
        
    plt.plot(model.entropy,'r-',linewidth=2)
    plt.xlabel('Number of epochs')
    plt.ylabel('Average cross-entropy loss')
    plt.savefig('Problem33_b.png')

 
    
    y_pred_train=model.predict_class(flatten_data(data["X_train"]))
    y_true_train=np.array([])
    for y in data['y_train']:
        y_true_train=np.append(y_true_train,np.argmax(y))
        
    ### Train accuracy
    train_acc=sum(y_true_train==y_pred_train)/len(y_true_train) 
    print("train accuracy: ", train_acc)
    

    y_pred_test=model.predict_class(flatten_data(data["X_test"]))
    y_true_test=np.array([])
    for y in data['y_test']:
        y_true_test=np.append(y_true_test,np.argmax(y))
    print(y_true_test)

    ## Test accuracy
    sum(y_true_test==y_pred_test)/len(y_true_test) 
    test_acc=sum(y_true_test==y_pred_test)/len(y_true_test) 
    print("Test accuracy: ",test_acc)
    
    
    return y_true_train,y_pred_train
   
    pass
    
    
def Problem34a(eta, num_epochs, hidden_layers):
    X_train_flat = flatten_data(data["X_train"])
    X_test_flat = flatten_data(data["X_test"])
    
    # Re-scale the features to have mean 0 and unit variance
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    ## TODO: Initialize and fit your neural network.
    train_entropy=[]
    test_entropy=[]
    for i in hidden_layers:
        model=NeuralNetwork(784,i,10,lamb=1e-4)
        model.fit(flatten_data(data["X_train"]),data['y_train'],eta=1e-4,num_epochs=50,Xtest=flatten_data(data["X_test"]),y_test=data["y_test"],test=True)

        train_entropy.append(model.entropy)
        test_entropy.append(model.entropy_test)
                            

    for i in range(0,len(train_entropy)):
        plt.plot(np.arange(1,len(train_entropy[i])+1),np.array(train_entropy)[i],linewidth=2,label=f'Hidden layers = {hidden_layers[i]}')
        plt.legend()
        plt.xlabel('Training epochs')
        plt.ylabel('Train set cross-entropy loss')
        
    plt.savefig('Train_loss.png')
    plt.show()
    

        
    for i in range(0,len(test_entropy)):
        plt.plot(np.arange(1,len(test_entropy[i])+1),np.array(test_entropy)[i],linewidth=2,label=f'Hidden layers = {hidden_layers[i]}')
        

    plt.legend()
    plt.xlabel('Training epochs')
    plt.ylabel('Test set cross-entropy loss')
    plt.savefig('Test_loss.png')


def Problem34b(eta, num_epochs, hidden_layers):
    X_train_flat = flatten_data(data["X_train"])
    X_test_flat = flatten_data(data["X_test"])
    
    # Re-scale the features to have mean 0 and unit variance
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    ## TODO: Initialize and fit your neural network.
    train_entropy=[]
    test_entropy=[]
    for i in eta:
        model=NeuralNetwork(784,hidden_layers,10,lamb=1e-4)
        model.fit(flatten_data(data["X_train"]),data['y_train'],eta=i,num_epochs=50,Xtest=flatten_data(data["X_test"]),y_test=data["y_test"],test=True)

        train_entropy.append(model.entropy)
        test_entropy.append(model.entropy_test)
                            

    for i in range(0,len(train_entropy)):
        plt.plot(np.arange(1,len(train_entropy[i])+1),np.array(train_entropy)[i],linewidth=2,label=f'eta = {eta[i]}')
        plt.legend()
        plt.xlabel('Training epochs')
        plt.ylabel('Train set cross-entropy loss')
        
    plt.savefig('Train_loss_eta.png')
    plt.show()
    

        
    for i in range(0,len(test_entropy)):
        plt.plot(np.arange(1,len(test_entropy[i])+1),np.array(test_entropy)[i],linewidth=2,label=f'eta = {eta[i]}')
        

    plt.legend()
    plt.xlabel('Training epochs')
    plt.ylabel('Test set cross-entropy loss')
    plt.savefig('Test_loss_eta.png')

    

    
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams.update({'font.size': 15})

def plot_confusion_matrix(y_true, y_pred):
    ''' Saves a confusion matrix plot for the true labels in y_true, and predicted labels in y_pred.
        The generated plot is saved to "p3_cm.pdf".
    
    Inputs:
        - y_true: a 1D (N,) array containing the true integer class labels for each example
        - y_pred: a 1D (N,) array containing the predicted integer class labels for each example
    '''
    
    cm = confusion_matrix(y_true, y_pred)
    ## Set the diagonals equal to -1 to aid visualization
    for i in range(10):
        cm[i, i] = -1
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig('p3_cm.pdf')
    
    
def Problem33d(y_true_train,y_pred_train):
    ind=np.where(y_true_train==4)  # All indices where y_treue=4

    misclass_ind=np.where(y_pred_train[ind]==9)    
    
    for i in range(0,10):
        print(y_true_train[ind[0][misclass_ind[0][i]]],y_pred_train[ind[0][misclass_ind[0][i]]])
    fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10)) = plt.subplots(2, 5, figsize=(15, 8))
    ax1.imshow(data['X_train'][ind[0][misclass_ind[0][0]]])
    ax2.imshow(data['X_train'][ind[0][misclass_ind[0][1]]])
    ax3.imshow(data['X_train'][ind[0][misclass_ind[0][2]]])
    ax4.imshow(data['X_train'][ind[0][misclass_ind[0][3]]])
    ax5.imshow(data['X_train'][ind[0][misclass_ind[0][4]]])
    ax6.imshow(data['X_train'][ind[0][misclass_ind[0][5]]])

    ax7.imshow(data['X_train'][ind[0][misclass_ind[0][6]]])
    ax8.imshow(data['X_train'][ind[0][misclass_ind[0][7]]])
    ax9.imshow(data['X_train'][ind[0][misclass_ind[0][8]]])
    ax10.imshow(data['X_train'][ind[0][misclass_ind[0][9]]])

    fig.savefig('Problem33d_4true.png')


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    