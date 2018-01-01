
# coding: utf-8

# In[65]:

import pandas as pd
import time


# In[66]:

def data_preprocessing(len_num):
    
    df = pd.read_csv("~/Desktop/adult-data.csv")
    
    #convert nominal value to numerical value 
    df["education"] = df.education.astype("category").cat.codes
    df["workclass"] = df.workclass.astype("category").cat.codes
    df["marital"] = df.marital.astype("category").cat.codes
    df["occupation"] = df.occupation.astype("category").cat.codes
    df["relationship"] = df.relationship.astype("category").cat.codes
    df["race"] = df.race.astype("category").cat.codes
    df["sex"] = df.sex.astype("category").cat.codes
    df["country"] = df.country.astype("category").cat.codes
    df["status"] = df.status.astype("category").cat.codes
    
    # convert numbers to bins (age)
    bins = list(range(10,91,10))
    group_names = list(range(0,8,1))
    df["age_c"] = pd.cut(df["age"],bins,labels=group_names)

    # convert numbers to bins (fnlwgt)
    bins = list(range(0,1500001,300000))
    group_names = list(range(0,5,1))
    df["fnlwgt_c"] = pd.cut(df["fnlwgt"],bins,labels=group_names)

    # convert numbers to bins (capital-gain)
    bins = [-1,0,20000,40000,60000,80000,100000]
    group_names = list(range(0,6,1))
    df["capitalgain_c"] = pd.cut(df["capital-gain"],bins,labels=group_names)

    # convert numbers to bins (capital-loss)
    bins = [-1,0,1000,2000,3000,4000,5000]
    group_names = list(range(0,6,1))
    df["capitalloss_c"] = pd.cut(df["capital-loss"],bins,labels=group_names)

    # convert numbers to bins (hours)
    bins = [0,20,40,60,80,100]
    group_names = list(range(0,5,1))
    df["hours_c"] = pd.cut(df["hours"],bins,labels=group_names)
    
    # change numbers to discrete but close numbers like items in basket
    data_dic = {0:"age_c",1:"workclass",2:"fnlwgt_c",3:"education",4:"marital",
                5:"occupation",6:"relationship",7:"race",8:"sex",9:"capitalgain_c",
                10:"capitalloss_c",11:"hours_c",12:"country",13:"status"}
    
    number_begin = 1
    new_name_list = []
    for i in range(14):
        name = data_dic[i]
        a = df[name].unique()
        b = max(a)
        bins = list(range(-1,b+1,1))
        group_names = list(range(number_begin,number_begin+b+1,1))
        new_name = name + "_c"
        new_name_list.append(new_name)
        df[new_name] = pd.cut(df[name],bins,labels=group_names)
        number_begin += len(a)
    
    number_begin -= 1
    
    # change dataframe to a list
    df["value"] = df[new_name_list].values.tolist()
    data_list = []
    for i in range(len_num):
        data_list.append(df["value"][i])
        #print(df["value"][i])
    
    return data_list,number_begin


# In[67]:

def apriori_gen(gen_list,length):
    result_list = []
    for i in range(len(gen_list)-1):
        for j in range(i+1,len(gen_list)):
            if length > 1:
                a_list = gen_list[i][:length-1]
                b_list = gen_list[j][:length-1]
                if a_list == b_list:
                    new_list = gen_list[i] + gen_list[j][-1:]
                    if has_infrequent_subset(new_list,gen_list) == False:
                        result_list.append(new_list) 
            elif (length == 1):
                new_list = gen_list[i] + gen_list[j]
                #print(new_list)
                result_list.append(new_list)
    
    result_list.sort()
    return result_list


# In[68]:

def has_infrequent_subset(new_list, gen_list):
    for i in range(len(new_list)):
        sub_list = new_list[:i] + new_list[i+1:]
        find = False
        #print(sub_list)
        for j in range(len(gen_list)):
            if set(sub_list) == set(gen_list[j]):
                find = True
                break
        if find == False:
            return True
    return False


# In[69]:

def apriori_scan(gen_list,data_list,min_support):
    result_list = []
    # count the potential frequent pattern
    scan = {}
    for i in range(len(data_list)):
        # change every row in data_list to a set in order to make comparison quicker
        data_set = frozenset(data_list[i])
        for j in range(len(gen_list)):
            now_set = frozenset(gen_list[j])
            if now_set.issubset(data_set):
                if now_set in scan.keys():
                    scan[now_set] += 1
                else:
                    scan[now_set] = 1
    
    for key in scan.keys():
        if scan[key] >= min_support:
            a = list(key)
            a.sort()
            result_list.append(a)
            # support[key] = scan[key]
    
    result_list.sort()
    return result_list


# In[70]:

def generate_c1(data_list):
    item_count = {}
    for i in range(len(data_list)):
        for j in range(len(data_list[i])):
            a = data_list[i][j]
            if a not in item_count.keys():
                item_count[a] = 1
            else:
                item_count[a] += 1
                
    return item_count


# In[71]:

def main_func(min_support,len_num):

    data_list,total_number = data_preprocessing(len_num)

    starttime = time.time()

    count = generate_c1(data_list)

    # support = {}
    candidate = []
    candidate.append([])
    fp = []

    for i in range(1,total_number+1,1):
        if i in count.keys():
            if count[i] >= min_support:
                a_list = []
                a_list.append(i)
                candidate[0].append(a_list)
                # support[frozenset(a_list)] = count[i]
    
    fp.append(candidate[0])

    k = 0

    while len(fp[k])!=0:
        k += 1
        # to generate candidate list and check it
        candidate.append(apriori_gen(fp[k-1],k))
        fp.append(apriori_scan(candidate[k],data_list,min_support))

    for i in range(len(fp)):
        if len(fp[i]) != 0:
            print(i+1)
        for j in range(len(fp[i])):
            print(fp[i][j])
        print("\n")

    endtime = time.time()

    time_cost = endtime - starttime
    
    return time_cost


# In[72]:

main_func(10000,20000)

