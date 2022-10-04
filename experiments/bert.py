# BERT

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import transformers
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW

from tqdm.notebook import tqdm
import math

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix

tqdm.monitor_interval = 0

############################### Tokenization and Embedding Generation ##############################

# Import BERT model and tokenizer
class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        print('Tokenizer:', self.tokenizer)
    
    def tokenize(self, sentences):
        print('Tokenizing messages...')
        tokens = self.tokenizer(
            sentences,
            max_length = 512, 
            padding = 'max_length',
            truncation = True,
            return_token_type_ids = False)
        return tokens        
    
    def convertToTensor(self, tokens):
        print('Converting to tensor...')
        seq = torch.tensor(tokens['input_ids'])
        mask = torch.tensor(tokens['attention_mask'])
        return seq, mask



# Embeddings Generator
class EmbeddingsGenerator():
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using:', self.device)
        print('CUDA available:', torch.cuda.is_available())
        print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')
        self.bert = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        self.batch_size = 32
        self.freezeParameters()

    def generateEmbedding(self, tokens):
        seq, mask = tokens
        del tokens
        quant_total = len(seq)
        print('Total seq: ', quant_total)
        quant_batches = math.ceil(quant_total/ self.batch_size)
        print('Number of batches: ', quant_batches)
        todos = list()

        pbar = tqdm(total=quant_batches, desc='Generating embeddings')
        for i in range(0, quant_batches):
            _min = i * self.batch_size
            _max = _min + self.batch_size
            tmp_seq = seq[_min:_max].to(self.device)
            tmp_mask = mask[_min:_max].to(self.device)
            _, embeddings = self.bert(tmp_seq, tmp_mask, return_dict=False)
            embeddings = embeddings.detach().cpu().numpy()
            todos.extend(embeddings)
            pbar.update(1)

        del tmp_seq
        del tmp_mask
        del embeddings
        del seq
        del mask
        return np.asarray(todos)

    def freezeParameters(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        print('Frozen BERT parameters...')


# Tokenize messages with BERT
def tokenizeMessagesBERT(messages):
  tokenizer = Tokenizer()
  tokens = tokenizer.tokenize(list(messages))
  tokens = tokenizer.convertToTensor(tokens)
  return tokens


# Generate embeddings with BERT
def generateEmbeddingsBERT(tokens):
  embeddingsGenerator = EmbeddingsGenerator()   
  tqdm.monitor_interval = 0
  embeddings = embeddingsGenerator.generateEmbedding(tokens).tolist()    
  return embeddings


############################### Classification ##############################

# Model architecture
class Arch(nn.Module):
  def __init__(self):
      super(Arch, self).__init__()
      # Dropout Layer
      self.dropout = nn.Dropout(0.1)
      # ReLu Layer
      self.relu =  nn.ReLU()
      # Dense Layer 1
      self.fc1 = nn.Linear(768, 512)
      # Dense Layer 2
      self.fc2 = nn.Linear(512, 2)
      # Softmax Activation Function
      self.softmax = nn.LogSoftmax(dim = 1)     

  def forward(self, entry):
      # Layer 1: 
      x = self.fc1(entry)
      # Layer 2: ReLu
      x = self.relu(x)
      # Layer 3: Dropout
      x = self.dropout(x)
      # Layer 4: Output Layer
      x = self.fc2(x)
      # Layer 5: Softmax
      x = self.softmax(x)
      
      return x


# Function to train the model
def train(device, model, train_dataloader, cross_entropy, optimizer):
  model.train() 
  total_loss = 0
  total_preds = []
  total_labels = []
  #pbar = tqdm(total=len(train_dataloader), desc='Training')
  
  for step, batch in enumerate(train_dataloader):
      # Put the batch on the chosen device (GPU)
      batch = [r.to(device) for r in batch]
      sent, labels = batch

      # Clears the previously calculated gradients
      model.zero_grad()

      # Get the model predictions
      preds = model(sent)

      # Calculates the error value between the forecast and the actual value
      loss = cross_entropy(preds, labels.to(torch.int64))

      # Total error sum
      total_loss += loss.item()

      loss.backward() # backward pass to calculate the gradients
      
      # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      
      # Update parameters
      optimizer.step()
      
      # Brings predictions and labels to CPU
      preds = preds.detach().cpu().numpy()
      labels = labels.detach().cpu().numpy() 
      
      total_preds.append(preds)
      total_labels.append(labels)
      #pbar.update(1)
  
  # Calculate average error
  avg_loss = total_loss / len(train_dataloader)

  total_preds  = np.concatenate(total_preds, axis = 0)
  total_labels  = np.concatenate(total_labels, axis=0)

  return avg_loss, total_preds, total_labels


# Function to evaluate the model
def evaluate(device, model, val_dataloader, cross_entropy):
  # Enters Evaluation mode (disables Dropout layer automatically)
  model.eval()
  total_loss = 0
  total_preds = []
  total_labels = []
  #pbar = tqdm(total=len(val_dataloader), desc='Evaluating')
  
  for step, batch in enumerate(val_dataloader):
      batch = [t.to(device) for t in batch]    
      sent, labels = batch
      with torch.no_grad(): # deactivate autograd       
          preds = model(sent) # model predictions          
          
          # Calculates the error between what is predicted and what is actually
          loss = cross_entropy(preds, labels.to(torch.int64))     

          # Total error sum
          total_loss += loss.item()
          
          # Store predictions and labels in CPU
          preds = preds.detach().cpu().numpy()  
          labels = labels.detach().cpu().numpy()  

          total_preds.append(preds)  
          total_labels.append(labels)
          #pbar.update(1)
  
  # Calculate the average error
  avg_loss = total_loss / len(val_dataloader)

  total_preds  = np.concatenate(total_preds, axis=0)
  total_labels  = np.concatenate(total_labels, axis=0)

  return avg_loss, total_preds, total_labels


# Convert to tensor
def convertToTensor(tokens, labels):
  seq = torch.tensor(tokens).float()
  y = torch.tensor(labels.tolist())
  return seq, y


# Create DataLoaders
def createDataloaders(x, y, batch_size):
  data = TensorDataset(x, y)
  sampler = RandomSampler(data)
  dataloader = DataLoader(data, sampler = sampler, batch_size = batch_size)
  return data, sampler, dataloader


# Create optimizer
def createOptimizer(model):
  optimizer = AdamW(model.parameters(), lr = 1e-4)
  return optimizer


# Create weights
def createWeights(device, train_labels):
  # Calculates the weights of each class
  class_wts = compute_class_weight(class_weight=None, classes=np.unique(train_labels), y=train_labels)
  
  # Convert weights to Tensors
  weights = torch.tensor(class_wts, dtype=torch.float)
  weights = weights.to(device)

  # Error Function
  cross_entropy  = nn.NLLLoss(weight=weights)
  return class_wts, weights, cross_entropy


# Classify with BERT
def fineTuneBERT(train_text, train_labels, val_text, val_labels, device, epochs, batch_size):
  train_seq, train_y = convertToTensor(train_text, train_labels)
  val_seq, val_y = convertToTensor(val_text, val_labels)

  train_data, train_sampler, train_dataloader = createDataloaders(train_seq, train_y, batch_size)
  val_data, val_sampler, val_dataloader = createDataloaders(val_seq, val_y, batch_size)

  # create the BERT model
  model = Arch().to(device)

  # create optimizer
  optimizer = createOptimizer(model)

  # create weights
  class_wts, weights, cross_entropy = createWeights(device, train_labels)
  cross_entropy.to(device)

  # set initial loss to infinite
  best_valid_loss = float('inf')
  best_epoch = 0

  # empty lists to store training and validation loss of each epoch
  train_losses=[]
  valid_losses=[]

  tqdm.monitor_interval = 0
  pbar = tqdm(total = epochs, desc = 'Epochs')

  for epoch in range(epochs):
    #print('\nEpoch {:} / {:}'.format(epoch + 1, epochs))

    train_loss, _, _ = train(device, model, train_dataloader, cross_entropy, optimizer)
    valid_loss, valid_preds, valid_labels = evaluate(device, model, val_dataloader, cross_entropy)

    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      best_epoch = epoch + 1
      torch.save(model.state_dict(), 'saved_weights.pt')

    #print(f'\nTraining Loss: {train_loss:.4f}')
    #print(f'Validation Loss: {valid_loss:.4f}')
    #print(f'Best Validation Loss: {best_valid_loss:.4f}')

    valid_preds = np.argmax(valid_preds, axis = 1)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    pbar.update(1)

  return valid_preds, best_epoch, best_valid_loss, model


# Test
def test(model, device, X_test, y_test):
  test_seq, test_y = convertToTensor(X_test, y_test)

  # loading best model
  path = 'saved_weights.pt'
  model.load_state_dict(torch.load(path))

  # get predictions for test data
  with torch.no_grad():
    preds = model(test_seq.to(device))
    preds = preds.detach().cpu().numpy()

  # model's performance
  preds = np.argmax(preds, axis = 1)

  return preds

