
# coding: utf-8

# In[172]:

import pandas as pd
import time


# In[173]:

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


# In[174]:

class node:
    def __init__(self, numvalue, numcount, parentnode):
        self.value = numvalue
        # record the frequency of the itemset(equal to the path from the root to the node)
        self.count = numcount
        # point to another node with same value
        self.nextnode = None
        self.parent = parentnode
        self.children = {}
        
    def increase(self, numcount):
        self.count = self.count + numcount


# In[175]:

def Tree(dataset,min_support):
    Table = {}
    
    for row in dataset.keys():
        for item in row:
            if item in Table.keys():
                #dataset is a dictionary，key is transaction information，value means its count 
                Table[item] += dataset[row]
            else:
                Table[item] = dataset[row]
    
    new_table = {}
    for key in Table.keys():
        #print(key,Table[key])
        if Table[key] >= min_support:
            new_table[key] = Table[key]
    
    if len(new_table.keys()) == 0:
        return None, None

    # to record the node with same value in the tree
    for key in new_table.keys():
        new_table[key] = [new_table[key], None]
    
    
    retree = node("no node",0,None)

    # scan the data set and build fp_tree
    for row, count in dataset.items():
        itemcount = {}
        for item in row:
            if item in new_table.keys():
                itemcount[item] = new_table[item][0] 
        if len(itemcount) > 0:
            # the number appears most will be constructed first
            ordereditems = [a[0] for a in sorted(itemcount.items(),key=lambda p:p[1], reverse=True)]
            #print(ordereditems)
            updatetree(ordereditems, retree, new_table, count)
    
    return retree, new_table


# In[176]:

def updatetree(items, currentree, table, count):
    if items[0] in currentree.children.keys():
        currentree.children[items[0]].increase(count)
    else:
        currentree.children[items[0]] = node(items[0],count,currentree)
        if table[items[0]][1] == None:
            table[items[0]][1] = currentree.children[items[0]]
        else:
            now = table[items[0]][1]
            while now.nextnode!= None:
                now = now.nextnode
            now.nextnode = currentree.children[items[0]]
    
    if len(items) > 1:
        updatetree(items[1:],currentree.children[items[0]],table,count)


# In[177]:

def createSet(dataSet):
    Dict = {}
    for i in range(len(dataSet)):
        if frozenset(dataSet[i]) not in Dict.keys():
            Dict[frozenset(dataSet[i])] = 1
        else:
            Dict[frozenset(dataSet[i])] += 1
    return Dict


# In[178]:

def findprefix(treenode,prefixpath):
    if treenode.parent != None:
        prefixpath.append(treenode.value)
        findprefix(treenode.parent, prefixpath)


# In[179]:

def findprepath(treenode):
    new_dataset = {}
    
    while treenode != None:
        prefixpath = []
        findprefix(treenode, prefixpath)
        if len(prefixpath) > 1:
            new_dataset[frozenset(prefixpath[1:])] = treenode.count
        treenode = treenode.nextnode
    
    return new_dataset


# In[180]:

def minetree(currentree, table, min_support, prefix, freqitemlist,max_length):
    
    if table != None:
        new_itemset = [a[0] for a in sorted(table.items(),key=lambda p:p[1][0])]
    
        for base in new_itemset:
            newfreqset = prefix.copy()
            newfreqset.add(base)
            newfreqset_list = list(newfreqset)
            newfreqset_list.sort()
            if len(newfreqset_list) > max_length:
                max_length = len(newfreqset_list)
            freqitemlist.append(newfreqset_list)
            new_dataset = findprepath(table[base][1])
            new_tree,new_table = Tree(new_dataset,min_support)

            if new_table != None:
                max_length = minetree(new_tree, new_table, min_support, newfreqset, freqitemlist,max_length)
    
    return max_length


# In[181]:

def main_func(min_support,len_num):
    
    data_list,total_number = data_preprocessing(len_num)
    data_changed = createSet(data_list)
    #print(data_changed)

    starttime = time.time()

    myfptree, myheadtable = Tree(data_changed, min_support)

    freqitems = []

    max_length = minetree(myfptree, myheadtable, min_support, set(), freqitems,0)

    freqitems.sort()

    for i in range(1,max_length+1,1):
        print(i)
        for j in freqitems:
            if len(j) == i:
                print(j)
        print("\n")

    endtime = time.time()

    time_cost = endtime - starttime
    
    return time_cost


# In[182]:

main_func(10000,20000)

