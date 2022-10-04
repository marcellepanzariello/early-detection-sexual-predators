# Utils

import pandas as pd
import datetime
import re

import shutil, sys
import glob
import os


# Read file with sexual predators labels from training dataset and returns IDs
def getSexualPredatorsTrainingDataset(folder):
  labelsFile = open(folder + 'pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt', 'r')
  content = labelsFile.readlines()
  sexualPredators = []

  for line in content:
      id = line.split('\n')
      sexualPredators.append(id[0])

  return sexualPredators


# Read file with sexual predators labels from test dataset and returns IDs
def getSexualPredatorsTestDataset(folder):
  labelsFile = open(folder + 'pan12-sexual-predator-identification-groundtruth-problem1.txt', 'r')
  content = labelsFile.readlines()
  sexualPredators = []

  for line in content:
      id = line.split('\n')
      sexualPredators.append(id[0])

  return sexualPredators


# Returns the DataFrame with the 10 conversations with more words
def getConversationsWithMoreWords(df):
  aux = df.copy()
  aux['words'] = aux['messages'].str.count(' ') + 1

  return aux[['conversation_id', 'words', 'messages']].sort_values(by=['words'], ascending = False).head(10)


# Remove repeated words in sequence from a text
def removeRepeatedWordsInSequence(text):
  m = re.search(r'\b(.+)\s+\1\b', text)

  if m:
    while m:
      text = re.sub(r'\b(.+)\s+\1\b', r'\1', text)      
      m = re.search(r'\b(.+)\s+\1\b', text)

  return text


# Group messages by conversations for strategies 1 and 3
def groupMessagesByConversations(connection, df, conversationSize):
  dfGroupedConversations = df.groupby("conversation_id")
  dfFirstMessages = dfGroupedConversations.head(conversationSize).reset_index(drop=True)
  dfFirstMessages.to_sql('conversations_filtered', connection, if_exists='replace', index=False)

  query = '''SELECT conversation_id, count(distinct(author)) as count_authors, group_concat(message, ' ') as messages
            FROM conversations_filtered 
            GROUP BY conversation_id;'''

  dfGroupedMessages = pd.read_sql(query, connection)

  c = connection.cursor()
  c.execute("DROP TABLE conversations_filtered")
  connection.commit()

  # Getting predatory conversation IDs
  dfPredatoryConversations = df[df['predatory_conversation'] == True]
  idPredatoryConversations = list(set(dfPredatoryConversations['conversation_id'].values))

  # Adding column "predatory_conversation" with value False initially
  # When the conversation is a predatory conversation, the label will be changed to True
  dfGroupedMessages = dfGroupedMessages.assign(predatory_conversation = False)

  for predatory_id in idPredatoryConversations:
    dfGroupedMessages.loc[(dfGroupedMessages['conversation_id'] ==  predatory_id), ['predatory_conversation']] = True
  
  return dfGroupedMessages


# Group messages by author for strategies 2 and 3
def groupMessagesByAuthor(connection, df, conversationSize):
  dfGroupedConversations = df.groupby("conversation_id")
  dfFirstMessages = dfGroupedConversations.head(conversationSize).reset_index(drop=True)
  dfFirstMessages.to_sql('conversations_filtered', connection, if_exists='replace', index=False)

  query = '''SELECT conversation_id, author, group_concat(message, ' ') as messages
            FROM conversations_filtered 
            GROUP BY conversation_id, author;'''

  dfGroupedMessages = pd.read_sql(query, connection)

  c = connection.cursor()
  c.execute("DROP TABLE conversations_filtered")
  connection.commit()

  # Getting predators IDs
  dfPredators = df[df['predator'] == True]
  idPredators = list(set(dfPredators['author'].values))

  # Adding column "predator" with value False initially
  # When the author is a sexual predator, the label will be changed to True
  dfGroupedMessages = dfGroupedMessages.assign(predator = False)

  for predator_id in idPredators:
    dfGroupedMessages.loc[(dfGroupedMessages['author'] ==  predator_id), ['predator']] = True
  
  return dfGroupedMessages


# Get current datetime
def getCurrentDatetime():
  now = datetime.datetime.now()
  year = '{:02d}'.format(now.year)
  month = '{:02d}'.format(now.month)
  day = '{:02d}'.format(now.day)
  hour = '{:02d}'.format(now.hour)
  minute = '{:02d}'.format(now.minute)
  dateHour = '{}-{}-{}--{}-{}'.format(year, month, day, hour, minute)
  return dateHour


# Move results file to drive
def moveResultsToDrive(resultsFolder):
  pattern = '/content/2022-*'
  for file in glob.iglob(pattern, recursive=True):
      filename = os.path.basename(file)
      shutil.move(file, resultsFolder + filename)
      print('Moved:', file)

