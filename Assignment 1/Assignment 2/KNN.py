#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np


data = np.genfromtxt('BankNote_Authentication.csv', delimiter=',')


np.random.shuffle(data)
num_train = int(0.7 * data.shape[0])
train_data = data[:num_train, :]
test_data = data[num_train:, :]


train_mean = np.mean(train_data[:, :4], axis=0)
train_std = np.std(train_data[:, :4], axis=0)
train_data[:, :4] = (train_data[:, :4] - train_mean) / train_std
test_data[:, :4] = (test_data[:, :4] - train_mean) / train_std


def knn(k, train_data, test_instance):
  
  distances = np.sqrt(np.sum((test_instance[:4] - train_data[:, :4])**2, axis=1))
  
 
  nearest_neighbors = np.argsort(distances)[:k]
  
 
  nearest_labels = train_data[nearest_neighbors, 4]
  
  
  label_counts = np.bincount(nearest_labels.astype(int))
  
  
  if len(label_counts) > 1 and label_counts[0] == label_counts[1]:
    return 0
  else:
    return np.argmax(label_counts)


for k in range(1, 10):
  num_correct = 0
  for test_instance in test_data:
    prediction = knn(k, train_data, test_instance)
    if prediction == test_instance[4]:
      num_correct += 1

  accuracy = num_correct / test_data.shape[0]
  print(f"k value: {k}")
  print(f"Number of correctly classified instances: {num_correct} Total number of instances: {test_data.shape[0]}")
  print(f"Accuracy: {accuracy}")
  print()


# In[ ]:




