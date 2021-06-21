# -*- coding: utf-8 -*-

"""
Name:    Shubham Kumar Sharma	
Email:    shubhrock777@gmail.com


Created on Sun Jun 20 12:48:17 2021

@author: SHBHAM
"""



#importing dataset
from sklearn.datasets import load_iris
#invoking library
import pandas as pd
import seaborn as sns
import numpy as np

# save load_iris() sklearn dataset to iris
iris = load_iris()

# np.c_ is the numpy concatenate function

model_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= ['sepal_length_cm',"sepal_width_cm","petal_length_cm","petal_width_cm"] + ['species'])



####### details of the model_df columns save as model_df_details 

model_df_details =pd.DataFrame({"column name":model_df.columns,
                          "Data type": ['ratio','ratio','ratio','ratio','nominal'],
                          "col_details":['length  of sepals in cm, relevant',
                                         'Width of sepals in cm, relevant',
                                         'length of petals in cm, relevant',
                                         'Width of petals in cm, relevant',
                                         'Three species of Iris, relevant']})




#3.	model_df Pre-processing
#3.1 model_df Cleaning ,Feature Engineering, etc
          

model_df.head(10)
            

#details of df 
model_df.info()
model_df.describe()         



#model_df types        
model_df.dtypes


#checking for na value
model_df.isna().sum()
model_df.isnull().sum()      #########no na values

#checking unique value for each columns
model_df.nunique()             #species have 3 factor



"""	Exploratory Data Analysis (EDA): on continuous 
	Summary
	Univariate analysis
	Bivariate analysis """


EDA = pd.DataFrame({"mean": model_df.iloc[:,0:4].mean(),
      "median":model_df.iloc[:,0:4].median(),
      "mode":model_df.iloc[:,0:4].mode(),
      "standard deviation": model_df.iloc[:,0:4].std(),
      "variance":model_df.iloc[:,0:4].var(),
      "skewness":model_df.iloc[:,0:4].skew(),
      "kurtosis":model_df.iloc[:,0:4].kurt()})

print(EDA)


# covariance for model_df set 
covariance = model_df.cov()
print(covariance)

# Correlation matrix 
corr = model_df.corr()
print(corr)   #########there is high correlation b/w petal_length_cm and petal_width_cm

###The heatmap is a data visualisation technique which is used to analyse the dataset as colors in two dimensions.
sns.heatmap(iris.corr(), linecolor = 'white', linewidths = 1)


######## count of unique value in species feature and unique value name
model_df.species.value_counts()
model_df.species.unique()


# Boxplot of independent variable distribution for each category of species

sns.boxplot(x = "species", y = "sepal_length_cm", data =model_df)
sns.boxplot(x = "species", y = "sepal_width_cm", data = model_df)
sns.boxplot(x = "species", y = "petal_length_cm", data = model_df)
sns.boxplot(x = "species", y = "petal_width_cm", data = model_df)


# Scatter plot for each categorical species of car
sns.stripplot(x = "species", y = "sepal_length_cm", jitter = True, data = model_df)
sns.stripplot(x = "species", y = "sepal_width_cm", jitter = True, data = model_df)
sns.stripplot(x = "species", y = "petal_length_cm", jitter = True, data =model_df)
sns.stripplot(x = "species", y = "petal_width_cm", jitter = True, data = model_df)



# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 

sns.pairplot(model_df, hue = "species") # With showing the category of each car model_dfyn in the scatter plot


#boxplot for every columns
model_df.boxplot(column=['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm'])   #some outlier present in sepal_width_cm (maybe Swamping False positive)





#######normalization all input features are in same unit so we dont need to normalize 

#############data prepration for model building 


#model_df = model_df.apply(pd.to_numeric)

#change dataframe to array
model_df_array = model_df.values

#split x and y (feature and target)
X = model_df_array[:,:4]
Y = model_df_array[:,4]


"""
SECTION 2 : Build and Train Model
Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 16 neuron, activation using ReLU
output layer : 3 neuron, represents the class of Iris, Softmax Layer
optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = default from keras.optimizer.SGD, 0.01
epoch = 200
"""
 
from sklearn.model_selection import train_test_split

#train test 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=7)




from tensorflow.keras import Sequential
from tensorflow.keras.layers import  Dense
from keras.utils import np_utils
# from keras.layers import Dropout,Flatten
import numpy as np



# one hot encoding outputs for  data sets 
y_train= np_utils.to_categorical( y_train)
y_test = np_utils.to_categorical(y_test)

n_classes= y_test.shape[1]

# Creating a user defined function to return the model for which we are
# giving the input to train the ANN mode
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(16,input_dim = 4,activation="relu"))
    model.add(Dense(n_classes,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
    return model
# building a cnn model using train data set and validating on test data set
model = design_mlp()
print(model.summary())

# fitting model on train data
history=model.fit(x_train, y_train, validation_split=0.3, shuffle=True, batch_size=16,epochs=200)


# Evaluating the model on test data  
#Confusion Matrix and Classification Report
from sklearn.metrics import confusion_matrix, classification_report 

# Prediction process of Final Model over test set
y_pred = model.predict(x_test)

# Classification Report
model_report = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print(model_report)

# Confusion Matrix
# multilabel-indicator is not supported so np.argmax should be used!
model_conf = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print(model_conf)

eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print ("Test Accuracy: %.3f%%" %(eval_score_test[1]*100)) 


# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Train Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
 


import matplotlib.pyplot as plt



# ##############train vs. test accuracy plot

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



################training vs test loss plot

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



###############Loss function plot

from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Process of plotting roc-auc curve belonging to all classes.

from itertools import cycle
from scipy import interp

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# Plot all ROC curves
lw = 2 # line_width
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
    
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Extending the ROC Curve to Multi-Class')
plt.legend(loc="lower right")
plt.show()




##################################################################################################




"""  If this function fails, the
process is repeated with a different AF, till the network learns
to approximate the ideal function. I """

# Multi Layer Perceptron Artificial Neural Network
from sklearn.neural_network import MLPClassifier 

# Setting up a primitive (non-validated) model
mlpc = MLPClassifier(random_state = 0)# ANN model object create


# Cross Validation Process
# Parameters for CV created in dictionary structure
# INFORMATION ABOUT THE INPUTED PARAMETERS
# alpha: float, default = 0.0001 L2 penalty (regularization term) parameter. (penalty parameter)


mlpc_params = {"alpha": [0.1, 0.01, 0.0001],
              "hidden_layer_sizes": [(8,12,16)],
              "solver" : ["lbfgs","adam","sgd"],
              "activation": ["relu","logistic","tanh"]}

from sklearn.model_selection import GridSearchCV

# Model CV process 
mlpc_cv_model = GridSearchCV(mlpc, mlpc_params,
                         cv = 5, # To make a 5-fold CV
                         n_jobs = -1, # Number of jobs to be run in parallel (-1: means to use all processors)
                         verbose = 2) # Controls the level of detail: higher means more messages gets value as integer.
mlpc_cv_model.fit(x_train, y_train) 


# The best parameter obtained as a result of CV process

print("The best parameters: " + str(mlpc_cv_model.best_params_))  

  
# Model Tuning
# Setting the Final Model with the best parameter

mlpc_tuned = mlpc_cv_model.best_estimator_

# Fitting Final Model
mlpc_tuned.fit(x_train, y_train)


# Prediction process of Final Model over test set
y_pred = mlpc_tuned.predict(x_test)

# Classification Report
model_report = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print(model_report)

# Confusion Matrix
# multilabel-indicator is not supported so np.argmax should be used!
model_conf = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print(model_conf)

 





###############################
#custom function for 
from math import exp



l_rate = 0.1
mu=0.001
n_epoch = 200
n_hidden = 1

# Calculate neuron activation for an input
def activate(weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
                activation += weights[i] * inputs[i]
        return activation
 
# Transfer neuron activation
def transfer(activation):
        return 1.0 / (1.0 + exp(-activation))
 
# Forward propagate input to a network output
def forward_propagate(network, row):
        inputs = row
        for layer in network:
                new_inputs = []
                for neuron in layer:
                        activation = activate(neuron['weights'], inputs)
                        neuron['output'] = transfer(activation)
                        new_inputs.append(neuron['output'])
                inputs = new_inputs
        return inputs
 
# Calculate the derivative of an neuron output
def transfer_derivative(output):
        return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
        for i in reversed(range(len(network))):
                layer = network[i]
                errors = list()
                if i != len(network)-1:
                        for j in range(len(layer)):
                                error = 0.0
                                for neuron in network[i + 1]:
                                        error += (neuron['weights'][j] * neuron['delta'])
                                errors.append(error)
                else:
                        for j in range(len(layer)):
                                neuron = layer[j]
                                errors.append(expected[j] - neuron['output'])
                for j in range(len(layer)):
                        neuron = layer[j]
                        neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
# Update network weights with error
def update_weights(network, row, l_rate):
        for i in range(len(network)):
                inputs = row[:-1]                
                if i != 0:
                        inputs = [neuron['output'] for neuron in network[i - 1]]
                for neuron in network[i]:
                        for j in range(len(inputs)):
                                temp = l_rate * neuron['delta'] * inputs[j] + mu * neuron['prev'][j]
                                
                                neuron['weights'][j] += temp
                                #print("neuron weight{} \n".format(neuron['weights'][j]))
                                neuron['prev'][j] = temp
                        temp = l_rate * neuron['delta'] + mu * neuron['prev'][-1]
                        neuron['weights'][-1] += temp
                        neuron['prev'][-1] = temp