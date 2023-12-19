# -*- coding: utf-8 -*-

import numpy as np
from numpynet.layer import Dense, ELU, ReLU, SoftmaxCrossEntropy
from numpynet.function import Softmax
from numpynet.utils import Dataloader, one_hot_encoding, load_MNIST, save_csv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

IntType = np.int64
FloatType = np.float64


class Model(object):
    """Model Your Deep Neural Network
    """
    def __init__(self, input_dim, output_dim):
        """__init__ Constructor

        Arguments:
            input_dim {IntType or int} -- Number of input dimensions
            output_dim {IntType or int} -- Number of classes
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_fn = SoftmaxCrossEntropy(axis=-1)
        self.labels = None
        self.build_model()

    def build_model(self):
        """build_model Build the model using numpynet API
        """
        self.hidden_layer_1 = Dense(self.input_dim, 256)
        self.hidden_layer_2 = Dense(256, 64)
        self.hidden_layer_3 = Dense(64, self.output_dim)
        self.elu1 = ELU(0.9)
        self.elu2 = ELU(0.9)
        self.elu3 = ELU(0.9)

    def __call__(self, X):
        """__call__ Forward propogation of the model

        Arguments:
            X {np.ndarray} -- Input batch

        Returns:
            np.ndarray -- The output of the model. 
                You can return the logits or probits, 
                which depends on the way how you structure 
                the code.
        """
        out1 = self.hidden_layer_1.__call__(X)
        out2 = self.elu1.__call__(out1)
        out3 = self.hidden_layer_2.__call__(out2)
        out4 = self.elu2.__call__(out3)
        out5 = self.hidden_layer_3.__call__(out4)
        out6 = self.elu3.__call__(out5)
        out7 = self.loss_fn.__call__(out6,self.labels)
        y_pred = self.loss_fn.y_pred
#         print("brop:", y_pred)
        return out7, y_pred

    def bprop(self, logits, labels, istraining=True):
        """bprop Backward propogation of the model

        Arguments:
            logits {np.ndarray} -- The logits of the model output, 
                which means the pre-softmax output, since you need 
                to pass the logits into SoftmaxCrossEntropy.
            labels {np,ndarray} -- True one-hot lables of the input batch.

        Keyword Arguments:
            istraining {bool} -- If False, only compute the loss. If True, 
                compute the loss first and propagate the gradients through 
                each layer. (default: {True})

        Returns:
            FloatType or float -- The loss of the iteration
        """
        if(istraining):
            grad1 = self.loss_fn.bprop()
            grad2 = self.elu3.bprop()
            grad3 = self.hidden_layer_3.bprop(grad2*grad1)
            grad4 = self.elu2.bprop()
            grad5 = self.hidden_layer_2.bprop(grad4*grad3)
            grad6 = self.elu1.bprop()
            grad7 = self.hidden_layer_1.bprop(grad6*grad5)

    def update_parameters(self, lr):
        """update_parameters Update the parameters for each layer.

        Arguments:
            lr {FloatType or float} -- The learning rate
        """
        self.hidden_layer_1.update(lr)
        self.hidden_layer_2.update(lr)
        self.hidden_layer_3.update(lr)


def train(model,
          train_X,
          train_y,
          val_X,
          val_y,
          max_epochs=200,
          lr=0.2,
          batch_size=16,
          metric_fn=accuracy_score,
          **kwargs):
    """train Train the model

    Arguments:
        model {Model} -- The Model object
        train_X {np.ndarray} -- Size: (60000, 784) -- Training dataset
        train_y {np.ndarray} -- Size: (60000,) -- Training labels
        val_X {np.ndarray} -- Size: (10000, 784) -- Validation dataset
        val_y {np.ndarray} -- Size: (10000,) -- Validation labels

    Keyword Arguments:
        max_epochs {IntType or int} -- Maximum training expochs (default: {20})
        lr {FloatType or float} -- Learning rate (default: {1e-3})
        batch_size {IntType or int} -- Size of each mini batch (default: {16})
        metric_fn {function} -- Metric function to measure the performance of 
            the model (default: {accuracy_score})
    """
    one_hot_train_y = one_hot_encoding(train_y)
    one_hot_val_y = one_hot_encoding(val_y)
    train_dataloader = Dataloader(X=train_X, y=one_hot_train_y, batch_size=batch_size, shuffle=True)
    val_dataloader = Dataloader(X=val_X, y=one_hot_val_y, batch_size=batch_size, shuffle=False)

    print('Training Started...')
    train_acc_all = []
    train_loss_all = []
    val_acc_all = []
    val_loss_all = []
    for j in tqdm(range(0,max_epochs)):
        train_acc = []
        train_loss_batch = 0
        
        
        for i, (features, labels) in enumerate(train_dataloader):
            model.labels = labels
            loss,y_pred = model.__call__(features)
            train_loss_batch += loss
            model.bprop(y_pred,labels, istraining = True) ## bprop needs logits, labels
            model.update_parameters(lr)            
            y_pred_one = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
            train_acc.append(metric_fn(labels, y_pred_one))
        
        
        train_loss_all.append(train_loss_batch/len(train_X))
        train_acc_all.append(sum(train_acc)/len(train_acc))
        
        if(j % 2 == 0):
            val_acc = []
            val_loss_batch = 0
            val_acc, val_loss, y_pred_val = valInference(model, val_X, val_y)
            val_acc_all.append(val_acc)
            val_loss_all.append(val_loss)

    return model, train_acc_all, train_loss_all, val_acc_all, val_loss_all


def valInference(model, X, y, batch_size=16, metric_fn=accuracy_score, **kwargs):
    """valinference Run the inference on the given dataset

    Arguments:
        model {Model} -- The Neural Network model
        X {np.ndarray} -- Size: (10000, 784) -- The dataset input
        y {np.ndarray} -- Size: (10000,) -- The dataset labels
        batch_size

    Keyword Arguments:
        batch_size {IntType or int} -- Size of each mini batch (default: {16})
        metric {function} -- Metric function to measure the performance of the model 
            (default: {accuracy_score})

    Returns:
        tuple of (float, float, list(or flattened numpy array)): A tuple of the accuracy, loss, and predicted labels
    """

    one_hot_test_y = one_hot_encoding(y)
    val_dataloader = Dataloader(X=X, y=one_hot_test_y, batch_size=batch_size, shuffle=False)
    
    test_acc = []
    y_pred_all = []
    loss_all = []
    test_loss_batch = 0
    for i, (features, labels) in enumerate(val_dataloader):
        ## forward pass
        model.labels = labels
        loss, y_pred = model.__call__(features)
        test_loss_batch = test_loss_batch + loss
        loss_all.append(test_loss_batch)
        y_pred_one = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
        y_pred_all.append([np.where(r==1)[0][0] for r in y_pred_one])
        test_acc.append(metric_fn(labels,y_pred_one))
    
    loss_ = test_loss_batch/len(X)
    acc_per_epoch_test = sum(test_acc)/len(test_acc)
    y_pred_all_flatten = np.array([item for sublist in y_pred_all for item in sublist])
    return acc_per_epoch_test, loss_, y_pred_all_flatten


def inference(model, X, batch_size=16,  **kwargs):
    """inference Run the inference for the test dataset without labels

    Arguments:
        model {Model} -- The Neural Network model
        X {np.ndarray} -- Size: (10000, 784) -- The dataset input

    Keyword Arguments:
        batch_size {IntType or int} -- Size of each mini batch (default: {16})
        metric {function} -- Metric function to measure the performance of the model 
            (default: {accuracy_score})

    Returns:
        list(or flattened numpy array): The predicted labels
    """    
    one_hot_test_y = one_hot_encoding(np.random.randint(10, size=X.shape[0]))
    test_dataloader = Dataloader(X=X, y=one_hot_test_y, batch_size=batch_size, shuffle=False)
    y_pred_all =[]
    
    
    for i, (features, labels) in enumerate(test_dataloader):
        _,y_pred = model.__call__(features)
        y_pred_one = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
        y_pred_all.append([np.where(r==1)[0][0] for r in y_pred_one])
    y_pred_all_flatten = np.array([item for sublist in y_pred_all for item in sublist])
    return y_pred_all_flatten


def main():
    print('Loading Data...')
    train_X, train_y = load_MNIST(path ='dataset/',name="train")
    val_X, val_y = load_MNIST(path = 'dataset/', name="val")
    test_X = load_MNIST(path = 'dataset/', name="test")
    test_loss, test_acc = None, None
    print('Loading Data Complete...')
    
    model = Model(input_dim= 784 , output_dim= 10)
    model,train_acc, train_loss, val_acc, val_loss =  train(model,
      train_X,
      train_y,
      val_X,
      val_y,
      max_epochs=200,
      lr=0.2,
      batch_size=16,
      metric_fn=accuracy_score)
    
    plt.plot(np.linspace(0,199,200), train_loss, label = 'train')
    plt.plot(np.linspace(0,198,100), val_loss, label = 'validation')
    plt.ylabel('Loss')
    plt.xlabel('# Epoch')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

    plt.plot(np.linspace(0,199,200), train_acc, label = 'train')
    plt.plot(np.linspace(0,198,100), val_acc, label = 'validation')
    plt.ylabel('Accuracy')
    plt.xlabel('# Epoch')
    plt.legend()
    plt.savefig('accuracy_curve.png')
    plt.show()
    
    test_pred = inference(model, test_X, batch_size = 16)
    save_csv(test_pred)
    print(test_pred.shape)
    # Your code ends here

    # Inference on valdationset dataset
    val_acc, val_loss, val_pred = valInference(model, val_X, val_y, batch_size = 16)
    print("Val acc,",val_acc)

    # print("Validation loss: {0}, Validation Acc: {1}%".format(val_loss, 100 * val_acc))
    if val_acc > 0.95:
        print("Your model is well-trained.")
    else:
        print("You still need to tune your model")
        
if __name__ == '__main__':
    main()
