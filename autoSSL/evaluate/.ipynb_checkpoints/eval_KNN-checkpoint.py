import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
 
from torch.utils.data import DataLoader
from tqdm import tqdm
def embed(x, embedding_model, device):
        embedding_model.eval()
        embedding_model.to(device)
        x = x.float().to(device)   # remove the unsqueeze operation
        return embedding_model(x).detach().cpu().numpy()

def eval_KNN(pipe_data, embedding_model, device='cuda', split=None, test=None):
    '''
    This function trains a KNN model and evaluates its accuracy on a test dataset.

    Parameters:
    pipe_data (PipeDataset): The whole data as a PipeDataset object
    embedding_model (torch.nn.Module): Model to create embeddings
    device (str): Device to perform computations on. Default is 'cuda' if available.
    split (float, optional): The ratio of samples to include in the train split.
    test (PipeDataset, optional): The test data as a PipeDataset object

    Returns:
    float: The accuracy of the model on the test data
    '''

    # Use split parameter to divide data into train and test if test is not provided
    if split is not None and test is None:
        train_data, test_data = pipe_data.split(split)
    else:
        train_data = pipe_data
        test_data = test

    # Extract features and labels from train and test data
    print("Load the training dataset to array")
    X_train, y_train = train_data.array[0], train_data.array[1]
    print("Load the testing dataset to array")
    X_test, y_test = test_data.array[0], test_data.array[1]

    # Get the embeddings for train and test data
    X_train_embedding = []
    print("embedding the training dataset")
    for x in tqdm(DataLoader(X_train, batch_size=128)):
        X_train_embedding.append(embed(x, embedding_model, device))
    X_train_embedding = np.concatenate(X_train_embedding)
    print("embedding the test dataset")
    X_test_embedding = []
    for x in tqdm(DataLoader(X_test, batch_size=128)):
        X_test_embedding.append(embed(x, embedding_model, device))
    X_test_embedding = np.concatenate(X_test_embedding)

    # Initialize the KNN Classifier
    knn = KNeighborsClassifier()
    print("Training in downstream")
    # Fit the model on the training data
    knn.fit(X_train_embedding, y_train)

    # Use the trained model to predict labels for the test data
    X_test_predicted = knn.predict(X_test_embedding)
    accuracy = accuracy_score(y_test, X_test_predicted)
    return accuracy




import os
import glob
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from autoSSL.models.Backbone import pipe_backbone
from sklearn.metrics import confusion_matrix
import seaborn as sns

def embed(x, embedding_model, device):
    embedding_model.eval()
    embedding_model.to(device)
    x = x.float().to(device)

    with torch.no_grad():
        embeddings = embedding_model(x)
        pooled_embeddings = torch.nn.functional.adaptive_avg_pool2d(embeddings, (1, 1))

    return pooled_embeddings.view(pooled_embeddings.size(0), -1).cpu().numpy()

def eval_everything(pipe_data, models, device='cuda', split=None, test=None, baseline=None):
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
    elif isinstance(models, list):
        models = {'name': ['model_'+str(i) for i in range(len(models))], 'model': models, 'address': None}
     
    baselines=[]
    results = []
    baselines_name=[]
    if baseline:
        for base in baseline:
            baseline_backbone, _ = pipe_backbone(backbone=base)
            models['model'].append(baseline_backbone)
            models['name'].append('baseline_' + base)
            baselines_name.append('baseline_' + base)
            baselines.append(baseline_backbone)
        
    writer = pd.ExcelWriter(models['address'].replace('.csv', '_confusion.xlsx'))

    for i, embedding_model in enumerate(tqdm(models['model'])):
        if embedding_model in baselines:
            pass
        else:
            embedding_model=embedding_model.backbone
        X_train_embedding = [embed(x, embedding_model, device) for x in DataLoader(X_train, batch_size=16)]
        X_train_embedding = np.concatenate(X_train_embedding)

        X_test_embedding = [embed(x, embedding_model, device) for x in DataLoader(X_test, batch_size=16)]
        X_test_embedding = np.concatenate(X_test_embedding)

        if X_test_embedding is None:
            accuracy = 'model_collapse'
            confusion = None
        else:
            clf = SGDClassifier(loss='log_loss')

            clf.fit(X_train_embedding, y_train)

            # Get class probabilities for each sample
            class_probs = clf.predict_proba(X_test_embedding)

            # Get the top 1 predictions
            top1_preds = np.argmax(class_probs, axis=1)
            top3_preds = np.argpartition(class_probs, -3, axis=1)[:,-3:]

            # Calculate confusion matrix
            confusion = confusion_matrix(y_test, top1_preds)
            # Normalize confusion matrix by row (i.e by the number of samples in each class)
            confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
            # Average accuracy is the mean of the diagonal elements (the correctly classified instances)
            top1_average_accuracy = np.mean(np.diag(confusion))
            
            # Calculate accuracy
            top1_accuracy = accuracy_score(y_test, top1_preds)
            top3_accuracy = np.mean([1 if y in top3 else 0 for y, top3 in zip(y_test, top3_preds)])

            # K-Nearest Neighbors classifier
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train_embedding, y_train)
            knn_preds = knn.predict(X_test_embedding)
            knn_accuracy = accuracy_score(y_test, knn_preds)

            accuracy = {
                "Top-1 Accuracy": top1_accuracy,
                "Top-3 Accuracy": top3_accuracy,
                "Top-1 Average Accuracy": top1_average_accuracy,
                "KNN Top-1 Accuracy": knn_accuracy  # KNN accuracy
            }

        namee = models["name"][i]
        results.append((namee, accuracy))

        
        del embedding_model
        torch.cuda.empty_cache()
        
        # Save confusion matrix to Excel file
        if confusion is not None:
            df_confusion = pd.DataFrame(confusion)
            df_confusion.to_excel(writer, sheet_name=namee)

    writer.save()

    if models['address'] is not None:
        df = pd.read_csv(models['address'])

        # If baselines are present, add new rows in the dataframe for them
        if baselines:
            for base, base_name in zip(baselines, baselines_name):
                # Initialize a new row with default values
                new_row = {col: None for col in df.columns}
                # Update the values we know
                new_row.update({
                    'dir_name': base_name,
                })
                # Append the new row to the dataframe
                df = df.append(new_row, ignore_index=True)

        # Assuming the results are in the same order as the models in the dataframe
        df['linear_top1_accuracy'] = [result[1]["Top-1 Accuracy"] for result in results]
        df['linear_top3_accuracy'] = [result[1]["Top-3 Accuracy"] for result in results]
        df['linear_top1_average_accuracy'] = [result[1]["Top-1 Average Accuracy"] for result in results]
        df['linear_knn_top1_accuracy'] = [result[1]["KNN Top-1 Accuracy"] for result in results]

        df.to_csv(models['address'], index=False)

    return results
