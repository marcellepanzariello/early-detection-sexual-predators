# Classifiers implementation

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np
import itertools

from time import time
from utils import *
from bert import *


###################### Naive Bayes Bernoulli ######################

def naiveBayesBernoulli(X_train, y_train):
  model = BernoulliNB().fit(X_train, y_train)
  return model


###################### Naive Bayes Multinomial ######################

def naiveBayesMultinomial(X_train, y_train):
  model = MultinomialNB().fit(X_train, y_train)
  return model


###################### Random Forest ######################

def randomForest(X_train, y_train):
  model = RandomForestClassifier(n_estimators=150, max_depth=None)
  model.fit(X_train, y_train)
  return model


###################### KNN ######################

def knn(X_train, y_train):
  model = KNeighborsClassifier(n_neighbors=5)
  model.fit(X_train, y_train)
  return model


###################### SVM ######################

def svm(X_train, y_train):
  model = SVC(kernel='linear')
  model.fit(X_train, y_train)
  return model


###################### Neural Network MLP ######################

def neuralNetworkMLP(X_train, y_train):
  model = MLPClassifier(hidden_layer_sizes=(100, 100, 100,), 
                        activation='relu', 
                        solver='adam',
                        max_iter=50, 
                        early_stopping=True,
                        validation_fraction=0.2)

  model.fit(X_train, y_train)
  return model


# Plot confusion matrix
def plotConfusionMatrix(cm):
    cmap = plt.get_cmap('Blues')
    
    plt.figure(figsize=(5, 3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xticks(np.arange(2), [True, False]) # rotation=45
    plt.yticks(np.arange(2), [True, False])

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, '{:,}'.format(cm[i, j]), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()


def matrix2(cm):
  import seaborn as sns
  import matplotlib.pyplot as plt     

  ax = plt.subplot(label="Label")
  sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Greens') # Blues ou rocket_r

  ax.set_xlabel('Predicted labels')
  ax.set_ylabel('True labels')
  ax.set_title('Confusion Matrix')
  ax.xaxis.set_ticklabels([True, False])
  ax.yaxis.set_ticklabels([True, False])


###################### Classification in training data ######################

def classify(classifier, X, y, messages, splits=10):
  accuracy_list = []
  precision_list = []
  recall_list = []
  f1_list = []
  f0_5_list = []
  kFold = StratifiedKFold(n_splits=splits)
  fold = 1

  indexes = []

  # Create results file
  resultsFile = open('{}--{}--{}m.txt'.format(getCurrentDatetime(), classifier, messages), 'x')

  for train_index, test_index in kFold.split(X, y):   
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    resultsFile.write('--------------------------------------------------------------------------\n')
    
    # Training
    t0 = time()
    if classifier == 'naive_bayes_bernoulli':
      resultsFile.write('NAIVE BAYES BERNOULLI - FOLD {}\n'.format(fold))
      resultsFile.write('--------------------------------------------------------------------------\n')
      model = naiveBayesBernoulli(X_train, y_train)
    else:
      if classifier == 'naive_bayes_multinomial':
        resultsFile.write('NAIVE BAYES MULTINOMIAL - FOLD {}\n'.format(fold))
        resultsFile.write('--------------------------------------------------------------------------\n')
        model = naiveBayesMultinomial(X_train, y_train)
      else:
        if classifier == 'random_forest':
          resultsFile.write('RANDOM FOREST - FOLD {}\n'.format(fold))
          resultsFile.write('--------------------------------------------------------------------------\n')
          model = randomForest(X_train, y_train)
        else:
          if classifier == 'knn':
            resultsFile.write('KNN - FOLD {}\n'.format(fold))
            resultsFile.write('--------------------------------------------------------------------------\n')
            model = knn(X_train, y_train)
          else:
            if classifier == 'svm':
              resultsFile.write('SVM - FOLD {}\n'.format(fold))
              resultsFile.write('--------------------------------------------------------------------------\n')
              model = svm(X_train, y_train)
            else:
              if classifier == 'neural_network_mlp':
                resultsFile.write('NEURAL NETWORK MULTI-LAYER PERCEPTRON - FOLD {}\n'.format(fold))
                resultsFile.write('--------------------------------------------------------------------------\n')
                model = neuralNetworkMLP(X_train, y_train)
              else:
                print('ERROR: Unknown classifier.')
                resultsFile.close()
                return
            

    train_time = time() - t0
    #resultsFile.write('TRAINING: {}\n'.format(train_index))
    #resultsFile.write('TEST: {}\n'.format(test_index))
    resultsFile.write('Shape of training data: {}\n'.format(X_train.shape))
    resultsFile.write('Shape of test data: {}\n'.format(X_test.shape))
    resultsFile.write('--------------------\n')
    resultsFile.write('Number of positives:\n')
    resultsFile.write('Train: {}\n'.format(len(y_train[y_train == True])))
    resultsFile.write('Test: {}\n'.format(len(y_test[y_test == True])))
    resultsFile.write('--------------------\n')
    resultsFile.write('Training time: %0.3fs\n' % train_time)

    # Test
    t0 = time()
    predictions = model.predict(X_test)
    test_time = time() - t0
    resultsFile.write('Test time:  %0.3fs\n' % test_time)
    resultsFile.write('--------------------\n')

    for i in range(len(predictions)):
      if predictions[i] == True:
        #print('Prediction at position (i): ', i)
        #print('test_index = ', test_index[i])
        
        index = test_index[i]
        indexes.append(index)

    #print(indexes)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    accuracy_list.append(accuracy)
    resultsFile.write('Accuracy: %0.4f\n' % accuracy)

    # Precision
    precision = precision_score(y_test, predictions)
    precision_list.append(precision)
    resultsFile.write('Precision: %0.4f\n' % precision)

    # Recall
    recall = recall_score(y_test, predictions)
    recall_list.append(recall)
    resultsFile.write('Recall: %0.4f\n' % recall)

    # F1
    f1 = f1_score(y_test, predictions)
    f1_list.append(f1)
    resultsFile.write('F1: %0.4f\n' % f1)

    # F0.5
    f0_5 = fbeta_score(y_test, predictions, beta=0.5)
    f0_5_list.append(f0_5)
    resultsFile.write('F0.5: %0.4f\n' % f0_5)

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels=[True, False])
    resultsFile.write(str(cm))
    resultsFile.write('\n')

    fold += 1

  accuracy_list = np.array(accuracy_list)
  precision_list = np.array(precision_list)
  recall_list = np.array(recall_list)
  f1_list = np.array(f1_list)
  f0_5_list = np.array(f0_5_list)

  # Calculate the averages
  metrics = dict()
  metrics['accuracy_avg'] = np.mean(accuracy_list, axis=0)
  metrics['precision_avg'] = np.mean(precision_list, axis=0)
  metrics['recall_avg'] = np.mean(recall_list, axis=0)
  metrics['f1_avg'] = np.mean(f1_list, axis=0)
  metrics['f0_5_avg'] = np.mean(f0_5_list, axis=0)
  
  metrics['accuracy_std'] = np.std(accuracy_list)
  metrics['precision_std'] = np.std(precision_list) 
  metrics['recall_std'] = np.std(recall_list)
  metrics['f1_std'] = np.std(f1_list)
  metrics['f0_5_std'] = np.std(f0_5_list)

  resultsFile.write('---------------------------------\n')
  resultsFile.write('Average Accuracy: %0.4f\n' % metrics['accuracy_avg'])
  resultsFile.write('Standard deviation - Accuracy:  %0.4f\n' % metrics['accuracy_std'])
  resultsFile.write('---------------------------------\n') 

  resultsFile.write('Average Precision: %0.4f\n' % metrics['precision_avg'])
  resultsFile.write('Standard deviation - Precision:  %0.4f\n' % metrics['precision_std'])
  resultsFile.write('---------------------------------\n') 

  resultsFile.write('Average Recall: %0.4f\n' % metrics['recall_avg'])
  resultsFile.write('Standard deviation - Recall:  %0.4f\n' % metrics['recall_std'])
  resultsFile.write('---------------------------------\n')

  resultsFile.write('Average F1: %0.4f\n' % metrics['f1_avg'])
  resultsFile.write('Standard deviation - F1:  %0.4f\n' % metrics['f1_std'])
  resultsFile.write('---------------------------------\n') 

  resultsFile.write('Average F0.5: %0.4f\n' % metrics['f0_5_avg'])
  resultsFile.write('Standard deviation - F0.5:  %0.4f\n' % metrics['f0_5_std'])
  resultsFile.write('---------------------------------\n') 

  resultsFile.close()

  return indexes, metrics 


# Classify with BERT
def classifyWithBERT(X, y, messages, splits=10, device='cpu', epochs=10, batch_size=32):
  accuracy_list = []
  precision_list = []
  recall_list = []
  f1_list = []
  f0_5_list = []
  loss_list = []
  best_epochs_list = []
  kFold = StratifiedKFold(n_splits=splits)
  fold = 1

  indexes = []

  # Create results file
  resultsFile = open('{}--{}--{}m.txt'.format(getCurrentDatetime(), 'bert', messages), 'x')

  for train_index, test_index in kFold.split(X, y):   
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    resultsFile.write('--------------------------------------------------------------------------\n')
    resultsFile.write('BERT - FOLD {}\n'.format(fold))
    resultsFile.write('--------------------------------------------------------------------------\n')
    resultsFile.write('Using: {}\n'.format(device))
    
    # Training
    #resultsFile.write('TRAINING: {}\n'.format(train_index))
    #resultsFile.write('VALIDATION: {}\n'.format(test_index))
    resultsFile.write('Shape of training data: {}\n'.format(X_train.shape))
    resultsFile.write('Shape of validation data: {}\n'.format(X_test.shape))
    resultsFile.write('--------------------\n')
    resultsFile.write('Number of positives:\n')
    resultsFile.write('Train: {}\n'.format(len(y_train[y_train == True])))
    resultsFile.write('Validation: {}\n'.format(len(y_test[y_test == True])))
    resultsFile.write('--------------------\n')

    # Call BERT method
    predictions, best_epoch, best_valid_loss, model = fineTuneBERT(X_train, y_train, X_test, y_test, device, epochs, batch_size)      
    
    resultsFile.write('\n--------------------\n')
    resultsFile.write('Best Epoch: {}\n'.format(best_epoch))
    resultsFile.write('Best Validation Loss: %0.4f\n' % best_valid_loss)
    loss_list.append(best_valid_loss)
    best_epochs_list.append(best_epoch)

    predictions = test(model, device, X_test, y_test)

    for i in range(len(predictions)):
      if predictions[i] == True:
        #print('Prediction at position (i): ', i)
        #print('test_index = ', test_index[i])
        
        index = test_index[i]
        indexes.append(index)

    #print(indexes)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    accuracy_list.append(accuracy)
    resultsFile.write('Accuracy: %0.4f\n' % accuracy)

    # Precision
    precision = precision_score(y_test, predictions)
    precision_list.append(precision)
    resultsFile.write('Precision: %0.4f\n' % precision)

    # Recall
    recall = recall_score(y_test, predictions)
    recall_list.append(recall)
    resultsFile.write('Recall: %0.4f\n' % recall)

    # F1
    f1 = f1_score(y_test, predictions)
    f1_list.append(f1)
    resultsFile.write('F1: %0.4f\n' % f1)

    # F0.5
    f0_5 = fbeta_score(y_test, predictions, beta=0.5)
    f0_5_list.append(f0_5)
    resultsFile.write('F0.5: %0.4f\n' % f0_5)

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels=[True, False])
    resultsFile.write(str(cm))
    resultsFile.write('\n')

    fold += 1

  accuracy_list = np.array(accuracy_list)
  precision_list = np.array(precision_list)
  recall_list = np.array(recall_list)
  f1_list = np.array(f1_list)
  f0_5_list = np.array(f0_5_list)
  loss_list = np.array(loss_list)
  
  # Calculate the averages
  metrics = dict();
  metrics['loss_avg'] = np.mean(loss_list, axis=0)
  metrics['accuracy_avg'] = np.mean(accuracy_list, axis=0)
  metrics['precision_avg'] = np.mean(precision_list, axis=0)
  metrics['recall_avg'] = np.mean(recall_list, axis=0)
  metrics['f1_avg'] = np.mean(f1_list, axis=0)
  metrics['f0_5_avg'] = np.mean(f0_5_list, axis=0)

  metrics['loss_std'] = np.std(loss_list)
  metrics['accuracy_std'] = np.std(accuracy_list)
  metrics['precision_std'] = np.std(precision_list) 
  metrics['recall_std'] = np.std(recall_list)
  metrics['f1_std'] = np.std(f1_list)
  metrics['f0_5_std'] = np.std(f0_5_list)

  # List with the number of the best epochs
  metrics['best_epochs'] = best_epochs_list

  resultsFile.write('---------------------------------\n') 
  resultsFile.write('Average Validation Loss: %0.4f\n' % metrics['loss_avg'])
  resultsFile.write('Standard deviation - Validation Loss:  %0.4f\n' % metrics['loss_std'])
  resultsFile.write('---------------------------------\n') 

  resultsFile.write('Average Accuracy: %0.4f\n' % metrics['accuracy_avg'])
  resultsFile.write('Standard deviation - Accuracy:  %0.4f\n' % metrics['accuracy_std'])
  resultsFile.write('---------------------------------\n') 

  resultsFile.write('Average Precision: %0.4f\n' % metrics['precision_avg'])
  resultsFile.write('Standard deviation - Precision:  %0.4f\n' % metrics['precision_std'])
  resultsFile.write('---------------------------------\n') 

  resultsFile.write('Average Recall: %0.4f\n' % metrics['recall_avg'])
  resultsFile.write('Standard deviation - Recall:  %0.4f\n' % metrics['recall_std'])
  resultsFile.write('---------------------------------\n')

  resultsFile.write('Average F1: %0.4f\n' % metrics['f1_avg'])
  resultsFile.write('Standard deviation - F1:  %0.4f\n' % metrics['f1_std'])
  resultsFile.write('---------------------------------\n') 

  resultsFile.write('Average F0.5: %0.4f\n' % metrics['f0_5_avg'])
  resultsFile.write('Standard deviation - F0.5:  %0.4f\n' % metrics['f0_5_std'])
  resultsFile.write('---------------------------------\n') 

  resultsFile.close()

  return indexes, metrics



###################### Classification in test data ######################

def testClassify(classifier, X_train, y_train, X_test, y_test, messages):
  indexes = []

  # Create results file
  resultsFile = open('{}--{}--{}m.txt'.format(getCurrentDatetime(), classifier, messages), 'x')
  resultsFile.write('--------------------------------------------------------------------------\n')
  
  # Training
  t0 = time()
  if classifier == 'naive_bayes_bernoulli':
    resultsFile.write('NAIVE BAYES BERNOULLI\n')
    resultsFile.write('--------------------------------------------------------------------------\n')
    model = naiveBayesBernoulli(X_train, y_train)
  else:
    if classifier == 'naive_bayes_multinomial':
      resultsFile.write('NAIVE BAYES MULTINOMIAL\n')
      resultsFile.write('--------------------------------------------------------------------------\n')
      model = naiveBayesMultinomial(X_train, y_train)
    else:
      if classifier == 'random_forest':
        resultsFile.write('RANDOM FOREST\n')
        resultsFile.write('--------------------------------------------------------------------------\n')
        model = randomForest(X_train, y_train)
      else:
        if classifier == 'knn':
          resultsFile.write('KNN\n')
          resultsFile.write('--------------------------------------------------------------------------\n')
          model = knn(X_train, y_train)
        else:
          if classifier == 'svm':
            resultsFile.write('SVM\n')
            resultsFile.write('--------------------------------------------------------------------------\n')
            model = svm(X_train, y_train)
          else:
            if classifier == 'neural_network_mlp':
              resultsFile.write('NEURAL NETWORK MULTI-LAYER PERCEPTRON\n')
              resultsFile.write('--------------------------------------------------------------------------\n')
              model = neuralNetworkMLP(X_train, y_train)
            else:
              print('ERROR: Unknown classifier.')
              resultsFile.close()
              return
          

  train_time = time() - t0
  resultsFile.write('Shape of training data: {}\n'.format(X_train.shape))
  resultsFile.write('Shape of test data: {}\n'.format(X_test.shape))
  resultsFile.write('--------------------\n')
  resultsFile.write('Number of positives:\n')
  resultsFile.write('Train: {}\n'.format(len(y_train[y_train == True])))
  resultsFile.write('Test: {}\n'.format(len(y_test[y_test == True])))
  resultsFile.write('--------------------\n')
  resultsFile.write('Training time: %0.3fs\n' % train_time)

  # Test
  t0 = time()
  predictions = model.predict(X_test)
  test_time = time() - t0
  resultsFile.write('Test time:  %0.3fs\n' % test_time)
  resultsFile.write('--------------------\n')

  for i in range(len(predictions)):
    if predictions[i] == True:
      index = i
      indexes.append(index)

  # Accuracy
  accuracy = accuracy_score(y_test, predictions)
  resultsFile.write('Accuracy: %0.4f\n' % accuracy)

  # Precision
  precision = precision_score(y_test, predictions)
  resultsFile.write('Precision: %0.4f\n' % precision)

  # Recall
  recall = recall_score(y_test, predictions)
  resultsFile.write('Recall: %0.4f\n' % recall)

  # F1
  f1 = f1_score(y_test, predictions)
  resultsFile.write('F1: %0.4f\n' % f1)

  # F0.5
  f0_5 = fbeta_score(y_test, predictions, beta=0.5)
  resultsFile.write('F0.5: %0.4f\n' % f0_5)

  # Confusion Matrix
  cm = confusion_matrix(y_test, predictions, labels=[True, False])
  resultsFile.write(str(cm))
  resultsFile.write('\n')

  metrics = dict()
  metrics['accuracy'] = accuracy
  metrics['precision'] = precision
  metrics['recall'] = recall
  metrics['f1'] = f1
  metrics['f0_5'] = f0_5

  resultsFile.close()

  return indexes, metrics 



