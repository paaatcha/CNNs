# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com
"""
from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
import glob
import os
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2HSV, COLOR_BGR2Lab, COLOR_BGR2XYZ, COLOR_BGR2HLS, COLOR_BGR2YUV
import unidecode
import sys 
sys.path.append('/home/labcin/CODIGOS/utils/')

from utils_img import create_dirs, split_folders_train_test_val



def get_analysis (filepath, dropna=True):    
    pd_dataset = pd.read_csv(filepath)   
    if (dropna):        
        # Tirando qualquer linha que possua um NaN
        pd_dataset = pd_dataset.dropna()
    
    feat_names = ['Cocou', 'Cresceu', 'Doeu', 'Mudou', 'Sangrou', 'Relevo', 'Idade']
    pd_dataset[feat_names] = pd_dataset[feat_names].replace(['N','S'], [0,1])
        
    print (pd_dataset.head())

'''
    This function just counts the amount of samples for each diagnostic
    Input:
        filepath: the csv filepath
        feat_names: if you have images and extra features, you may indicate in this
                    parameter. If it is None, the function load only the image and lavels
    Output:
        pd_dataset: a pandas dataframe with the diagnostic and the # of samples
'''
def get_count_dataset (filepath, dropna=True):    
    pd_dataset = pd.read_csv(filepath)   
    if (dropna):        
        # Tirando qualquer linha que possua um NaN
        pd_dataset = pd_dataset.dropna()
        
    return pd_dataset.groupby(['Diagnostico'])['Path'].count()

'''
    Auxiliary function to format the labels by removing some characters
    
    Inputs:
        x: a string or a list of string
    Outputs:
        x: the formatted string
'''
def format_labels (x):
    def format_string_lab (x):
        x = unidecode.unidecode(x.decode('utf-8')).replace(' ', '_').lower()
        x_labs = x.split('_')
        if (len(x_labs) > 1):
            x = '_'.join(x_labs[:-1])
        else:
            x = x_labs[0]
        return x

    if (type(x) == list):
        for i in range(len(x)):
            x[i] = format_string_lab(x[i])        
    else:
        x = format_string_lab(x)
        
    return x
    
    
'''
    This function reads a csv file and convert all input to a python dictionary. 
    It may take into account images, extra features and the labels
    Input:
        filepath: the csv filepath
        feat_names: if you have images and extra features, you may indicate in this
                    parameter. If it is None, the function load only the image and labels
        verbose: set it true if you'd like to see some prints on the screen        
    
    Output:
        dict_data: a python dictionary that has the images name as keys and the 
                   extra features and labes as values
        labels: a list with all labels for this dataset
'''
def get_dict_data (filepath, feat_names=None, valid_labels=None, verbose=False):
    dict_data = dict()
    pd_labels = pd.read_csv(filepath)   
    labels = list()
    
    # Formating valid labels if it exists
    valid_labels = format_labels(valid_labels)
    
    if (feat_names is not None):        
        # Tirando qualquer linha que possua um NaN
        pd_labels = pd_labels.dropna()
        
        # Fazendo N = 0 e S = 1
        pd_labels[feat_names] = pd_labels[feat_names].replace(['N','S'], [0,1])        
        scalar_feat = pd_labels.as_matrix(feat_names)
        cont = 0
    
    for x in pd_labels.Diagnostico.unique():
        x = format_labels (x)            
        labels.append(x)    
    
    for lin in pd_labels.iterrows():    
        diag = format_labels(lin[1].Diagnostico)   
        path = lin[1].Path
        
        if (valid_labels is not None):
            if (diag not in valid_labels):
                continue
        
        if (verbose):
            print (diag, " - ", path  )
        
        if (feat_names is not None):
            dict_data[path] = (scalar_feat[cont], diag)
            cont += 1
        else:
            dict_data[path] = (0, diag)
        
    return dict_data, labels

'''
    It receives an RGB image and convert it to the colorspace desired
    Input:
        img: the image path
        colorspace: the colorpace ('HSV', 'Lab', 'XYZ', 'HSL' or 'YUV')
    Output:
        img: the converted image 

'''
def convert_colorspace (img_path, colorspace):
    img = imread(img_path)
    if (colorspace == 'HSV'):
        img = cvtColor(img, COLOR_BGR2HSV)                   
    elif (colorspace == 'Lab'):
        img = cvtColor(img, COLOR_BGR2Lab)    
    elif (colorspace == 'XYZ'):
        img = cvtColor(img, COLOR_BGR2XYZ)    
    elif (colorspace == 'HLS'):
        img = cvtColor(img, COLOR_BGR2HLS)
    elif (colorspace == 'YUV'):
        img = cvtColor(img, COLOR_BGR2YUV)
    else:
        print ("There is no conversion for {}".format(colorspace))
        raise ValueError
    
    return img

'''
    It copies an group of images from filepath folder to a tree of folders based on the img labels.
    It split all images in labels folders and you may convert to a given colorspace
    
    Inputs:
        filepath: the imgs filepath
        dict_data: a python dictionary obtained from get_dict_data
        labels: a list contained all img labels. It is used to create the folder's tree
        colorspace: the desired colorspace
        name_main_folder: the tree's root folder
        verbose: set it true to see some prints on the screen        
'''
def cp_images (filepath, dict_data, labels, colorspace='RGB', name_main_folder='img_per_folders', verbose=False):
    paths = glob.glob(filepath)        
    N = len(paths)
    n = 0
    missing = list()

    # Creating a new dir tree
    create_dirs(name_main_folder, labels)    
    
    for p in paths:            
        n+=1
        if (verbose):
            print ('Working on img {} of {}'.format(n,N))            
        try:                
            name_img = p.split('/')[-1]            
            scalar_feat = dict_data[name_img][0]                
            label_img = dict_data[name_img][1]            
            new_path_feat = name_main_folder + '/' + label_img + '/' + name_img.split('.')[0] + '_feat.txt'
            
            if (colorspace=='RGB'):
                new_path_img = name_main_folder + '/' + label_img
                cp_command = 'cp ' + p + ' ' + new_path_img
                os.system(cp_command)
            else:
                img = convert_colorspace(p, colorspace)
                new_path_img = name_main_folder + '/' + label_img + '/' + name_img                
                imwrite (new_path_img, img)
                
            np.savetxt(new_path_feat, scalar_feat, fmt='%i', delimiter=',')
        except KeyError:
            if (verbose):
                print ("There is no {} in the dictionary".format(name_img))
            missing.append(p)
                
    if (verbose):
        print ("\n##################")
        print ("# of images: {}".format(N))
        print ("# of copied images: {}".format(N-len(missing)))
        print ("##################")
    


get_analysis("/home/labcin/AndrePacheco/Datasets/PAD/dataset.csv")

#feat = ['Cocou', 'Cresceu', 'Doeu', 'Mudou', 'Sangrou', 'Relevo', 'Idade']
#val_labs = ['CARCINOMA BASO CELULAR C80', 'CARCINOMA ESPINO CELULAR C44.9', 'CERATOACANTOMA D23', 'CERATOSE ACTINICA L57.0', 
#       'CERATOSE SEBORREICA L82', 'CORNO CUTANEO L75.8', 'LENTIGO MALIGNO D03.9', 'MELANOMA C43.9', 'NEVO MELANOCITICO D22.9']

#dados, labels = get_dict_data("/home/labcin/AndrePacheco/Datasets/PAD/dataset.csv", feat, valid_labels=val_labs)
#cp_images ('/home/labcin/AndrePacheco/Datasets/PAD/imgs/*.jpg', dados, val_labs, name_main_folder='/home/labcin/AndrePacheco/Datasets/PAD/img_per_folders', verbose=True)


#create_dirs('/home/labcin/AndrePacheco/Datasets/PAD/pad_menor_splited', )
#split_folders_train_test_val ('/home/labcin/AndrePacheco/Datasets/PAD/img_per_folders', '/home/labcin/AndrePacheco/Datasets/PAD/pad_menor_splited', scalar_feat='txt')






