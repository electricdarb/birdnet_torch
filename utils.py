import boto3
from config import KEY_ID, SECRET_KEY
import os
import datetime
import numpy as np
import torch

def make_runname(prefix = None):
    # create a runname from date
    if isinstance(prefix, str): # if the prefix is an isntance of string 
        return f'{prefix}_{datetime.datetime.now(datetime.timezone.utc).strftime("%y%m%d%H%M%S")}'
    else: 
        return datetime.datetime.now(datetime.timezone.utc).strftime("%y%m%d%H%M%S")

def save_model(runname, model, bucketname = 'bradfordgillbirddatabucket', foldername = 'modellogs'): 
    # save model locally and to cloud
    filename = f'{runname}.pth'
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    path = os.path.join(foldername, filename)

    session = boto3.Session(
            aws_access_key_id=KEY_ID,
            aws_secret_access_key=SECRET_KEY)
    s3 = session.resource('s3')

    torch.save(model.state_dict(), path)
    result = s3.Bucket(bucketname).upload_file(path, path)

def load_model(runname, bucketname = 'bradfordgillbirddatabucket', foldername = 'modellogs'): 
    # load model from local or cloud 

    filename = f'{runname}.pth'
    path = f'{foldername}/{filename}'

    if not os.path.exists(path):
        if not os.path.exists(foldername):
            os.mkdir(foldername)
        session = boto3.Session(
                aws_access_key_id=KEY_ID,
                aws_secret_access_key=SECRET_KEY)
        s3 = session.resource('s3')

        results = s3.Bucket(bucketname).download_file(path, path)

    return torch.load(path)

def log_history(runname, history, bucketname = 'bradfordgillbirddatabucket', foldername = 'historylogs'):
    filename = f'{runname}.npy'
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    path = os.path.join(foldername, filename)

    np.save(path, history.history)

    session = boto3.Session(
            aws_access_key_id=KEY_ID,
            aws_secret_access_key=SECRET_KEY)

    s3 = session.resource('s3')
    result = s3.Bucket(bucketname).upload_file(path, path)

def get_history(runname,  bucketname = 'bradfordgillbirddatabucket', foldername = 'historylogs'):
    filename = f'{runname}.npy'
    path = os.path.join(foldername, filename)

    if not os.path.exists(path):
        if not os.path.exists(foldername):
            os.mkdir(foldername)
        session = boto3.Session(
                aws_access_key_id=KEY_ID,
                aws_secret_access_key=SECRET_KEY)

        s3 = session.resource('s3')
        result = s3.Bucket(bucketname).download_file(path, path)

    return np.load(path, allow_pickle = True)

def get_class_names():
    path = 'cub200data/CUB_200_2011/classes.txt'
    names = []

    with open(path, 'r') as file:
        for line in file:
            names.append(line[line.find('.') + 1:])
    return names

def format_for_yolo():
    pass
            

if __name__ == '__main__':
    get_class_names()