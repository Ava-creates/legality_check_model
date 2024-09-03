import pickle
import random
import sys
# # Specify the path to your .pkl file
# file_path = "./data_old/commit/tiramisu/data_factory_kb4083/benchmarks_by_execution/processed/done/function_2mm_LARGE/output.txt"
# file_path = "dataset_batch550000-716507_train.pkl"

"""
computations ['A_out'] line Matrix Transform 4x4 {A_out, } [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1]

2d skewing only 
illegal schedules 
max matrice size if 3
read from the json to get the beter stuff 
no fusion?
"""
    
def get_schedule(input_string):
    keywords = ["<illegal>", "<legal>"]
    start_position = 0
    substrings_with_keywords = []
    while start_position < len(input_string):
    # Find the next occurrence of any keyword in the remaining part of the string
        next_keyword = None
        next_keyword_position = len(input_string)
    
        for keyword in keywords:
            keyword_position = input_string.find(keyword, start_position)
            if keyword_position != -1 and keyword_position < next_keyword_position:
                next_keyword = keyword
                next_keyword_position = keyword_position
    
        if next_keyword is not None:
        # Extract the substring before the next keyword
            substring_before = input_string[start_position:next_keyword_position]+next_keyword
        
        # Add the substring along with the corresponding keyword
            substrings_with_keywords.append(substring_before)
        
        # Update the start position for the next iteration
            start_position = next_keyword_position + len(next_keyword)
        else:
        # No more keywords found, add the remaining part of the string
            substrings_with_keywords.append(input_string[start_position:])
            break
            
    return  substrings_with_keywords     
            
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

def m_1d_transform(trans_str):
    trans = trans_str[trans_str.index("[")+1: trans_str.index("]")]
    
    trans= trans.split(",")
    
    trans_mat =[]
    
    for i in trans:
        trans_mat.append(int(i))
        
#     print(trans_mat)
    
    transform = {"reverse": [], "interchange": [] , "skewing":[]}
    
    if(trans_mat[0]==-1):
        transform["reverse"].append("L0")
    else:
        transform =None
        
    return transform
    
def m_2d_transform(trans_str):
    """Matrix Transform 2x2 {tmp_init, } [1, 0, 0, 1]"""
    
    trans = trans_str[trans_str.index("[")+1: trans_str.index("]")]
    
    trans= trans.split(",")
    
    trans_mat =[]
        
    for i in trans:
        trans_mat.append(int(i))

    transform = {"reverse": [], "interchange": [] , "skewing":[]}
#     print(trans_mat)
    count=0
    if (trans_mat[0] == -1):
        transform["reverse"].append("L0")
        count+=1
    
    if(trans_mat[3] ==-1):
        transform["reverse"].append("L1")
        count+=1
        
    if(trans_mat[0]==0 and trans_mat[1]==1):
        transform["interchange"].append("L0")
        count+=1
    
    if(trans_mat[3]==0 and trans_mat[2]==1):
        transform["interchange"].append("L1")
        count+=1
    if(trans_mat[1] >1 and trans_mat[0]>1):
        transform["skewing"].append(["L0",trans_mat[0] ])
        transform["skewing"].append(["L1", trans_mat[1]])
        count+=1
    if(count== 0):
        transform= None

    return transform 
    

    
def m_3d_transform(trans_str):
    
    trans = trans_str[trans_str.index("[")+1: trans_str.index("]")]
    
    trans= trans.split(",")
    trans_mat =[]
    for i in trans:
        trans_mat.append(int(i))
        
    transform = {"reverse": [], "interchange": [] , "skewing":[]}
    count =0
    
    if (trans_mat[0] == -1):
        transform["reverse"].append("L0")
        count+=1
    
    if(trans_mat[4] ==-1):
        transform["reverse"].append("L1")
        count+=1
        
           
    if(trans_mat[8] ==-1):
        transform["reverse"].append("L2")
        count+=1
        
    if(trans_mat[0]==0 and trans_mat[4]==0):
        transform["interchange"].append("L0")   #for interchange only 2 loops should be in the list
        transform["interchange"].append("L1")
        count+=1
        
    if(trans_mat[0]==0 and trans_mat[8]==0):
        transform["interchange"].append("L0")   #for interchange only 2 loops should be in the list
        transform["interchange"].append("L2")
        count+=1
    
    if( trans_mat[4]==0 and trans_mat[8]==0):
        transform["interchange"].append("L1")
        transform["interchange"].append("L2")
        count+=1
       
   ###confused about skewing 
    if(trans_mat[2]>=1):
        transform["skewing"].append(["L0",trans_mat[1]])
        transform["skewing"].append(["L1",trans_mat[2]])
        count+=1
 
    if(trans_mat[5] >=1):
        transform["skewing"].append(["L1", trans_mat[4]])
        transform["skewing"].append(["L2",trans_mat[5]])
        count+=1
        
    if(count==0):
        transform = None
    return transform

def m_4d_transform(trans_str):
    trans = trans_str[trans_str.index("[")+1: trans_str.index("]")]
    
    trans= trans.split(",")
    trans_mat =[]
    for i in trans:
        trans_mat.append(int(i))
        
    transform = {"reverse": [], "interchange": [] , "skewing":[]}
    count =0
    if (trans_mat[0] == -1):
        transform["reverse"].append("L0")
        count+=1
    
    if(trans_mat[5] ==-1):
        transform["reverse"].append("L1")
        count+=1
           
    if(trans_mat[10] ==-1):
        transform["reverse"].append("L2")
        count+=1
        
    if(trans_mat[15] ==-1):
        transform["reverse"].append("L2")
        count+=1
  
        
    if(trans_mat[0]==0 and trans_mat[5]==0):
        transform["interchange"].append("L0")   #for interchange only 2 loops should be in the list
        transform["interchange"].append("L1")
        count+=1
        
    if(trans_mat[0]==0 and trans_mat[10]==0):
        transform["interchange"].append("L0")   #for interchange only 2 loops should be in the list
        transform["interchange"].append("L2")
        count+=1
        
    if(trans_mat[0]==0 and trans_mat[15]==0):
        transform["interchange"].append("L0")   #for interchange only 2 loops should be in the list
        transform["interchange"].append("L3")
        count+=1
        
    if( trans_mat[5]==0 and trans_mat[10]==0):
        transform["interchange"].append("L1")
        transform["interchange"].append("L2")
        count+=1
   
    
    if( trans_mat[5]==0 and trans_mat[15]==0):
        transform["interchange"].append("L1")
        transform["interchange"].append("L3")
        count+=1
        
    if( trans_mat[10]==0 and trans_mat[15]==0):
        transform["interchange"].append("L1")
        transform["interchange"].append("L3")
        count+=1
   
   
   ###confused about skewing 
    if(trans_mat[2]>=1):
        transform["skewing"].append(["L0",trans_mat[1]])
        transform["skewing"].append(["L1",trans_mat[2]])
        count+=1
 
    if(trans_mat[5] >=1):
        transform["skewing"].append(["L1", trans_mat[5]])
        transform["skewing"].append(["L2",trans_mat[6]])
        count+=1
        
    if(trans_mat[10] >=1):
        transform["skewing"].append(["L2", trans_mat[10]])
        transform["skewing"].append(["L3",trans_mat[11]])
        count+=1
        
#     if(trans_mat[] >=1):
#         transform["skewing"].append(["L2", trans_mat[10]])
#         transform["skewing"].append(["L3",trans_mat[11]])
#         count+=1
        
        
    if(count==0):
        transform = None
    return transform
    
  
#it_dict is the iterator list for the computation 
def get_transform_list(mat, it_dict):
    li=[]
    
    for key in mat.keys():
        if(len(mat[key])>0):
            if(key== "reverse" ):
                for loop in mat[key]:
                    li.append([2,  0,0 , int(loop[1:]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0] )  #adding depth of loop not name
            elif(key =="interchange"):
                li.append([1,  int(mat[key][0][1:]), int(mat[key][1][1:]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0])  #again the depth of the two loops 
                it_dict[int(mat[key][0][1:])], it_dict[int(mat[key][1][1:])] = it_dict[int(mat[key][1][1:])],it_dict[int(mat[key][1][1:])]           

            elif(key == "skewing"):                
                c= len(mat[key])
    
                if(c==2):
                    a,b = linear_diophantine_default( mat[key][0][1], mat[key][1][1])
                    li.append([3, 0,0,0, int(mat[key][0][0][1:]), int(mat[key][1][0][1:]), 0, mat[key][0][1], mat[key][1][1],a,b,0,0,0,0,0])
                        
                else:
                    li.append([3, 0,0,0, int(mat[key][0][0][1:]), 0, 0, mat[key][0][1], 0,0,0,0,0,0,0,0])

            else:
                print("somethign definitely went wrong")
                print("transforms", mat)
                
    return li
    

def final_schedules(z, program_annotation, tree_structure):
    transform_list=[]
    
    for i in z:
        lis = get_schedule(i)

        for s in lis:
         if("<illegal>" in s):
            print("in illegal schedule")
#             count_of_illegal_schedules+=1
            transforms={}
            schedule=s.split("\n")

            '''
            {"tmp_init" : {"shiftings" : null,"tiling" : {},"unrolling_factor" : null,"parallelized_dim" : null, "transformations_list" : []}, "tmp_prod" : {"shiftings" : null,"tiling" : {},"unrolling_factor" : null,"parallelized_dim" : null, "transformations_list" : []}, "D_beta" : {"shiftings" : null,"tiling" : {},"unrolling_factor" : null,"parallelized_dim" : null, "transformations_list" : [[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ]}, "D_prod" : {"shiftings" : null,"tiling" : {},"unrolling_factor" : null,"parallelized_dim" : null, "transformations_list" : [[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ]}, "fusions" : null,
            '''
            trans_yay =["Matrix Transform", "Tiling", "Parallelize", "Unrolling"]
            for line in schedule:
                matches = [item for item in trans_yay if item in line]
                if( not matches):
                    continue
                   
                computations = line[line.index("{")+1:line.index("}")].split(",")
                computations = computations[: len(computations)-1]
#                 print("computations",computations, "line", line)
                computations = [item.replace(" ", "") for item in computations]

#                 print(program_annotation["computations"].keys())
                
                for computation in computations:
                    if(computation not in transforms.keys()):
                        transforms[computation] = {"shiftings" : None,"tiling" : {},"unrolling_factor" :None,"parallelized_dim" : None, "transformations_list" : []}
                    it_dict = program_annotation["computations"][computation]["iterators"].copy()  
                    if("Matrix Transform" in line):

                        t= None
                        if( "2x2" in line ):
                            trans_mat = m_2d_transform(line)
                            if(trans_mat == None):
                                continue
                                t = get_transform_list(trans_mat, it_dict)
                                for instance in t:
                                    transforms[computation]["transformations_list"] = instance
                        elif("3x3" in line):
                            trans_mat = m_3d_transform(line)
                            if(trans_mat == None):
                                continue
                            
                            t = get_transform_list(trans_mat, it_dict)
                            for instance in t:
                                    transforms[computation]["transformations_list"] = instance
                        elif("1x1" in line):
                            
                            trans_mat = m_1d_transform(line)
                            if(trans_mat == None):
                                continue
                            t = get_transform_list(trans_mat, it_dict)
                            for instance in t:
                                    transforms[computation]["transformations_list"] = instance
                                    
                        elif("4x4" in line):
                            trans_mat = m_4d_transform(line)
                            if(trans_mat == None):
                                continue
                            t = get_transform_list(trans_mat, it_dict)
                            for instance in t:
                                    transforms[computation]["transformations_list"] = instance
                                    
                            
                        else:
                            print("dimension of transformation matrix is more than 3 or it does not have any transformations")
                            print(line)
                            break

                    elif("Tiling" in line):
                        lin=line.split()[1:]
                        lin=lin[:lin.index("{")]
                        l=len(lin)
                        til={}
                        start=0
                        loops=[]
                        tiling_dims=[]
                        tiling_factors=[]
                        while( start < l):
#                             print(int(lin[start][1:]))
                            tiling_dims.append(int(lin[start][1:]))
                            tiling_factors.append(int(lin[start+1]))
                            loops.append(int(lin[start][1:]))
                            start+=2
                        
                        
                        transforms[computation]["tiling"] = {"tiling_depth": len(tiling_dims), "tiling_dims" : tiling_dims,"tiling_factors" : tiling_factors}
                        
#                         print(transforms[computation]["tiling"]) 
#                         sys.exit()

                        if(l==6 and len(loops)==3):
                                    first_dim = it_dict[loops[0]]
                                    second_dim =it_dict[loops[1]]
                                    third_dim = it_dict[loops[2]]

                                    index = it_dict.index(first_dim)
                                    it_dict[index: index + 1] = (
                                    first_dim + "_outer",
                                    second_dim + "_outer",
                                    third_dim + "_outer",
                                    )
                                    index = it_dict.index(second_dim)
                                    it_dict[index : index + 1] = (
                                    first_dim + "_inner",
                                    second_dim + "_inner",
                                    third_dim + "_inner",
                                    )
                                    it_dict.remove(third_dim)

                        elif(l==4):
                            a, b = it_dict[loops[0]], it_dict[loops[1]]
                            index = it_dict.index(a)
                            it_dict[index : index + 1] = a + "_outer", b + "_outer"
                            index = it_dict.index(b)
                            it_dict[index : index + 1] = a+ "_inner", b+ "_inner"

                        else:
                            print("FEREWFERWFERrfer")



                    elif("Parallelize" in line):
                        lin=line.split()[1:]
                        lin=lin[:lin.index("{")]
                        transforms[computation]["parallelized_dim"]= int(lin[0][1:])  #okay might have to deal with iterator list and shit 
                    elif("Unrolling" in line):
                        unroll ={}
                        lin=line.split()[1:]
                        lin=lin[:lin.index("{")]
                        #Unrolling L3 4 
                        transforms[computation]["unrolling_factor"]= int(lin[1])    
                        dim_index = len(it_dict) - 1
                        dim_name = it_dict[-1]
                        it_dict[dim_index : dim_index + 1] = (
                                dim_name + "_Uouter",
                                dim_name + "_Uinner",
    
                        )
            transforms["fusions"]= None                        
            transforms["tree_structure"] = tree_structure
            transforms["legality_check"] = 0 
            transforms["exploration_method"] = -1
            transforms["execution_times"] = []
            transform_list.append(transforms)
    return transform_list

# A function to extract the transformations applied on a spesific computation in the form of a vector of tags
# Padding is added if the number of transformations is less than the maximum value of MAX_NUM_TRANSFORMATIONS
# The tag representation is as follows:
    #         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'third_skewing_loop', 'skew_parameter_1', 'skew_parameter_2', 'skew_parameter_3', 'skew_parameter_4', 'skew_parameter_5', 'skew_parameter_6', 'skew_parameter_7', 'skew_parameter_8', 'skew_parameter_9']
    #     Where the type_of_transformation tag is:
    #         - 0 for no transformation being applied
    #         - 1 for loop interchange
    #         - 2 for loop reversal
    #         - 3 for loop skewing
    # In the case for skewing we are specifying the new values for the transformed submatrix           
            
def caller(file_path, program_annotation, tree_structure):
    lines=""
    with open(file_path, 'r') as f:
       lines = f.readlines()
    ughh=""
    for line in lines:
        ughh+=line
    count = ughh.count("<illegal>")
    print("illegal count" , count)
    z= ughh.split("Generated Halide IR:")
    z=z[1:]

        
    return final_schedules(z, program_annotation, tree_structure)
    