#!/usr/bin/env python
# coding: utf-8
import pickle
import random
import pandas as pd
import json
# # Specify the path to your .pkl file
file_path = "full_rl.pkl"
# file_path = "debug.pkl"
print("here")
    # Open the file in read binary model
with open(file_path, 'rb') as f:
   data = pickle.load(f)
        # Process or use the loaded data as needed
import random

legal_yes=0
legal_no=0
dropped=0
drop_list=[]
not_dropped=0
iterators=[]

def linear_diophantine_default(f_i,f_j):
    found = False
    gamma = 0
    sigma = 1
    if ((f_j == 1) or (f_i == 1)):
        gamma = f_i - 1
        sigma = 1
    else:
        if((f_j == -1) and (f_i > 1)):
            gamma = 1
            sigma = 0       
        else:     
            i =0
            while((i < 100) and (not found)):     
                if (((sigma * f_i ) % abs(f_j)) ==  1):
                            found = True
                else:
                    sigma+=1
                    i+=1
            if(not found):
                print('Error cannof find solution to diophantine equation')
                return
            gamma = ((sigma * f_i) - 1 ) / f_j
    
    return gamma, sigma


def get_child_list(j):

    iterators.append(j)
    l=[]
    w=[]
   
    for i in data[key]["program_annotation"]["iterators"][j]["child_iterators"]:
                dict={}
                dict["loop_name"]= i
                w.append(i)
                dict["computations_list"]=data[key]["program_annotation"]["iterators"][i]["computations_list"]
                dict["child_list"] = get_child_list(i)
                l.append(dict)            
    return l


def get_tree(key):
    tree_structure ={}
    tree_structure["roots"]=[]
    w=[]
    iterators=[]
    
    for j in data[key]["program_annotation"]["iterators"]:
                if(j in iterators):
                    break
                dict={}
                dict["loop_name"]= j
                dict["computations_list"]=data[key]["program_annotation"]["iterators"][j]["computations_list"]
                dict["child_list"] = get_child_list(j)
                tree_structure["roots"].append(dict)
                
    return tree_structure



def get_schedule_list(itr, key):
    global legal_yes
    global legal_no
    schedule_list=[]
    for j in data[key]["schedules_legality"]: #j is a schedule
#        try:
        dict1={}        
        bracket_split = j.split("{")
        fusions=[]
        fusions_json={}
        for i in bracket_split:   #schedule can have multiple computations  ading cimputations to schedules dict
              
                if(not i ):
                    continue
                it_dict=[]
                if( i[0]!='F'):
                    
                    lol=i[:i.index("}")]
                    dict1[lol]={"tiling":{}, "unrolling_factor": None,  'parallelized_dim': None, 'shiftings': None,'transformations_list': []}

                    
                    it_dict = data[key]["program_annotation"]["computations"][lol]["iterators"].copy()  #iterator list for the computation renewed per schedule as it gets changed during the transformations
                    if fusions:
                     for fusion in fusions:
                       if lol in fusion:
                        iterator_comp_name = fusion[0]
                        it_dict = data[key]["program_annotation"]["computations"][iterator_comp_name]["iterators"].copy()

#                 s= ""
#                 if("T" in i or "U" in i): 
#                    for index, ch in enumerate(i):
#                     if(ch =="I"):
#                         s=s+i[index : index+(i[index:].index(")")+1)]
#                    for index, ch in enumerate(i):
#                     if(ch =="R"):
#                         s+=i[index : index+i[index:].index(")")+1]
#                    for index, ch in enumerate(i):
#                     if(ch =="S"): 
#                         s+=i[index : index+i[index:].index(")")+1]
#                    for index, ch in enumerate(i):
#                     if(ch =="P"):
#                         s+=i[index : index+i[index:].index(")")+1]
#                    for index, ch in enumerate(i):
#                     if(ch =="T"):
#                         s+=i[index : index+i[index:].index(")")+1]
#                    for index, ch in enumerate(i):
#                     if(ch =="U"):
#                         s+=i[index : index+i[index:].index(")")+1]
                          
#                 if(i[0]=='F'):
                s=i
                                 
                flag=0
                
                for index, ch  in enumerate(s):   #for going through diffeent transformationss
                    
                    if ch=="S":

                        li=[]
                        c= s[index:(index+ s[index:].index(")"))].count("L")
                        if(c ==3):
                            w= s[index+2:]
                            comma_split= w[: w.index(")")].split(",")
                            li= [3, 0,0,0, int(comma_split[0][1:]), int(comma_split[1][1:]),int(comma_split[2][1:]), int(comma_split[3]), int(comma_split[4]),int(comma_split[5]),0,0,0,0,0,0]
                        
                        elif(c==2):
                            w= s[index+2:]
                            comma_split= w[: w.index(")")].split(",")
                            a,b = linear_diophantine_default(int(comma_split[2]), int(comma_split[3]))
                            li= [3, 0,0,0, int(comma_split[0][1:]), int(comma_split[1][1:]), 0, int(comma_split[2]), int(comma_split[3]),a,b,0,0,0,0,0]
                        
                        else:
                            w= s[index+2:]
                            comma_split= w[: w.index(")")].split(",")
                                
                            li= [3, 0,0,0, int(comma_split[0][1:]), 0, 0, int(comma_split[0]), 0,0,0,0,0,0,0,0]
                        dict1[lol]['transformations_list'].append(li)


                    if(ch == 'U'):
                            dim_index = len(it_dict) - 1
                            dim_name = it_dict[-1]
#                             print("uh from string unroll", s[index+3])
#                             print("from code", dim_index)
#                             print(it_dict)
                            dict1[lol]["unrolling_factor"] = s[index+5]
                            it_dict[dim_index : dim_index + 1] = (
                            dim_name + "_Uouter",
                            dim_name + "_Uinner",
                            )
                        
                    if(ch == 'P'):
                            dict1[lol]["parallelized_dim"] =it_dict[int(s[index+3])]
                                                           
                    if( ch =='I'):
#                            try:
                            li=[]
                            li= [1,  int(s[index+3]), int(s[index+6]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0]
#                             try:
                            it_dict[int(s[index+3])], it_dict[int(s[index+6])] = it_dict[int(s[index+6])],it_dict[int(s[index+3])]


                            dict1[lol]['transformations_list'].append(li)
                            
                    if( ch =='R'):
                            li=[]
                            li= [2,  0,0 , int(s[index+3]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0]
                            dict1[lol]['transformations_list'].append(li)
                        
                    tiling= {}
                    if (ch=="T"):
                            tiling_dims=[]
                            tiling_factors=[]
                            c =int(s[index+1])
                            if (c==1):
                                w= s[index+3:]
                                comma_split= w[: w.index(")")].split(",")
                                tiling_dims.append(it_dict[int(comma_split[0][1:])])
                                tiling_factors.append(comma_split[1])
                            elif(c==2):                      
                                 w= s[index+3:]
                                 comma_split= w[: w.index(")")].split(",")
                                
#                                 if(a not in ["i0", "i1", "i2", "i3"] or b not in ["i0", "i1", "i2", "i3"]):
#                                     print(a)
#                                     print(b)
#                                     print(s)
#                                     print(it_dict)
                                
                                 a, b = it_dict[int(comma_split[0][1:])], it_dict[int(comma_split[1][1:])]
          
                                 tiling_dims.append(it_dict[int(comma_split[0][1:])])
                                 tiling_dims.append(it_dict[int(comma_split[1][1:])])
                           
                                 tiling_factors.append(comma_split[2])
                                 tiling_factors.append(comma_split[3])
                               
                                 i = it_dict.index(a)
                                    
                                 it_dict[i : i + 1] = a + "_outer", b + "_outer"
                                 i = it_dict.index(b)
                                 it_dict[i : i + 1] = a+ "_inner", b+ "_inner"

                            elif(c==3):
                                w= s[index+3:]
                                comma_split= w[: w.index(")")].split(",")
                                
#                                 try:
                                tiling_dims.append(it_dict[int(comma_split[0][1:])])
                                tiling_dims.append(it_dict[int(comma_split[1][1:])])
                                tiling_dims.append(it_dict[int(comma_split[2][1:])])
#                                 except:
#                                  print(d)
#                                  print(s)
#                                  print(key)
                                tiling_factors.append(comma_split[3])
                                tiling_factors.append(comma_split[4])
                                tiling_factors.append(comma_split[5])
                                
                                
                                first_dim = it_dict[int(comma_split[0][1:])]
                                second_dim =it_dict[int(comma_split[1][1:])]
                                third_dim = it_dict[int(comma_split[2][1:])]
                  
                                i = it_dict.index(first_dim)
                                it_dict[i : i + 1] = (
                                first_dim + "_outer",
                                second_dim + "_outer",
                                third_dim + "_outer",
                                )
                                i = it_dict.index(second_dim)
                                it_dict[i : i + 1] = (
                                first_dim + "_inner",
                                second_dim + "_inner",
                                third_dim + "_inner",
                                )
                                it_dict.remove(third_dim)


                            else:
                            
                                print("woops greater than 3 tiling dumbass")
 
                            tiling = {'tiling_depth': int(s[index+1]), 'tiling_dims': tiling_dims, 'tiling_factors':tiling_factors}

                            dict1[lol]["tiling"]=tiling
        
                    if(ch =='F'):
                        flag=1
                        w= s[index+2:]
                        comma_split= w[: w.index(")")].split(",")
                        comma_split.append(len(comma_split))
                        fusions.append(comma_split)
                        fusions_json[comma_split[0]]= s
#                         print(fusions)
                        
                if(flag==0):
                    dict1["unfuse_iterators"]=[]
                if(flag==1):
                    dict1["fusions"]= fusions

                
        dict1["tree_structure"]=get_tree(key)
                
        dict1["legality_check"]= data[key]["schedules_legality"][j]
        
        if(dict1["legality_check"]==1):
            legal_yes+=1
        else:
            legal_no+=1
            

        dict1['sched_str']=j
        
        if(dict1["legality_check"] ==1 and legal_yes >=669456):
            continue

        schedule_list.append(dict1)

    return schedule_list


big_dataset={}
i =0
for key in data:
            
           try:
            big_dataset[key]={"filename":None, "node_name":None, "parameters":None,                                                                             "program_annotation":{"memory_size": data[key]["program_annotation"]["memory_size"],                                               "iterators": data[key]["program_annotation"]["iterators"],                                                                         "computations": data[key]["program_annotation"]["computations"]},
                              "initial_execution_time": 1, 
                              "schedules_list":get_schedule_list(data[key]["program_annotation"]["computations"], key),
                              "exploration_trace":None}
#             print("not exception")
#             if(legal_yes >= 669456 and legal_no >= 669450):
#                break
#             i+=1
#             if(i>500):
#                 break
           except Exception as e:
            continue
        
           
                  
print("done")   



from sklearn.model_selection import train_test_split

s = pd.Series(big_dataset)
training_data, rest_data = train_test_split(s, train_size=0.8)

# Split the rest into validation and test
validation_data, test_data = train_test_split(rest_data, test_size=0.5)
print(len(training_data.keys()))
print(len(validation_data))
print(len(test_data))


print("number of schedules dropped", dropped)
print("number of schedules",  not_dropped)

with open("train_converted_all_balanced_2024.pkl", "wb") as f:
        pickle.dump(training_data, f)
        
with open("val_converted_all_balanced_2024.pkl", "wb") as f:
       pickle.dump(validation_data, f)
        
with open("test_converted_all_balanced_2024.pkl", "wb") as f:
       pickle.dump(test_data, f)

# print("loading from the converted file")
# for key in data:
    
#     for j in data[key]["schedules_list"]:
#         if("F" in j["sched_str"]):
#             print(data[key]["schedules_list"])
    
print("dumped")
with open('drop_list_convert.txt', 'w') as fp:
    for item in drop_list:
        # write each item on a new line
        fp.write(str(item))
    print('Done writing')


