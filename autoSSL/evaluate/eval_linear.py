import os
import glob
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

def embed(x, embedding_model, device):
    embedding_model.eval()
    embedding_model.to(device)
    x = x.float().to(device)
    return embedding_model(x).detach().cpu().numpy()

def eval_linear(pipe_data, models, device='cuda', split=None, test=None):
    if split is not None:
        train_data, test_data = pipe_data.split(split)
    elif test is None:
        train_data = pipe_data
        test_data = pipe_data
    else:         
        train_data = pipe_data
        test_data = test

    print("Load the training and testing dataset")
    X_train, y_train = train_data.array[0], train_data.array[1]
    X_test, y_test = test_data.array[0], test_data.array[1]

    if isinstance(models, torch.nn.Module):
       models = {'name': ['model_0'], 'model': [models], 'address': None} 
    if isinstance(models, list):
        models = {'name': ['model_'+str(i) for i in range(0,len(models))], 'model': models, 'address': None}
   
    results = []

    for i, embedding_model in enumerate(tqdm(models['model'])):
        X_train_embedding = [embed(x, embedding_model, device) for x in DataLoader(X_train, batch_size=128)]
        X_train_embedding = np.concatenate(X_train_embedding)

        X_test_embedding = [embed(x, embedding_model, device) for x in DataLoader(X_test, batch_size=128)]
        X_test_embedding = np.concatenate(X_test_embedding)

        if X_test_embedding is None:
            accuracy = 'model_collapse'
        else:
            clf = SGDClassifier()
            clf.fit(X_train_embedding, y_train)

            # Get class probabilities for each sample
            class_probs = clf.predict_proba(X_test_embedding)

            # Get the top 1 and top 3 predictions
            top1_preds = np.argmax(class_probs, axis=1)
            top3_preds = np.argpartition(class_probs, -3, axis=1)[:,-3:]

            # Calculate accuracy
            top1_accuracy = accuracy_score(y_test, top1_preds)
            top3_accuracy = np.mean([1 if y in top3 else 0 for y, top3 in zip(y_test, top3_preds)])

            accuracy = {
                "Top-1 Accuracy": top1_accuracy,
                "Top-3 Accuracy": top3_accuracy
            }

        namee=models["name"][i]
        results.append((namee, accuracy))
        
    if models['address'] is not None:
        df = pd.read_csv(models['address'])
        df['linear_top1_accuracy'] = [result[1]["Top-1 Accuracy"] for result in results]
        df['linear_top3_accuracy'] = [result[1]["Top-3 Accuracy"] for result in results]
        df.to_csv(models['address'], index=False)
    
    return results  
