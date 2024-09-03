#!/usr/bin/env python
# coding: utf-8
import pickle
import random
import pandas as pd
import json
# # Specify the path to your .pkl file
file_path = "pickled/val_converted_all_balanced.pkl"
# file_path = "debug.pkl"
print("here")
    # Open the file in read binary model
with open(file_path, 'rb') as f:
   data = pickle.load(f)
        # Process or use the loaded data as needed
bigdataset={}
legal=0
legal_no=0
f=0
for key in data:
    
    j=data[key]["schedules_list"]
    
    for i in j:
        if(legal >=600000 and legal_no >574968):
                f=1
                break
        if(i["legality_check"] ==1 and legal>= 600000):
            l=1
            continue
        if(i["legality_check"] ==1):
            legal+=1
        else:
            legal_no+=1
            
    bigdataset[key]= data[key]      
    if(f==1):
        break
        
print(legal_no)
print(legal)
with open("val_converted_all_balanced_2024_balancepy.pkl", "wb") as f:
       pickle.dump(bigdataset, f)
            
        

            


