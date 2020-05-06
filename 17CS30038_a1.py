#####################################################################################
"""""
Roll Number - 17CS30038
Name - Yash Parag Butala
Assign - 1
numpy is used for array calculations
csv file handling is done using pandas(allowed by sir)
output print function is made using pprint and without it. 
this is a generalized code so can be used for other dataset
execution : python3 17CS30038_a1.py
"""
####################################################################################
import pandas   as pd
import numpy as  np
from pprint import  pprint
dataset = pd.read_csv('data1_19.csv', names=['pclass','age','gender','survived'])
print(dataset)
###################
def entropy(col):

    elements,counts = np.unique(col,return_counts = True)
    for i in range(len(elements)):
    	p1=(-counts[i]/np.sum(counts))
    	p2=np.log2(counts[i]/np.sum(counts))
    	entropy = np.sum([p1*p2])
    return entropy
################### 
    
###################
def Gain(data,split_attribute_name,target_name="survived"):

    total_entr = entropy(data[target_name])
    param=split_attribute_name
    vals,counts= np.unique(data[param],return_counts=True)

    Weighted_Ent = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[param]==vals[i]).dropna()[target_name]) for i in range(len(vals))])

    Info_Gain = total_entr 
    Info_Gain-=Weighted_Ent

    return Info_Gain
       
###################
def predict(query,tree,default = 0):

    for key in list(query.keys()):
        if key in list(tree.keys()):
           
            try:	
                result = tree[key][query[key]] 
            except:
                return default
            res=tree[key][query[key]]           	
            if not(isinstance(result,dict)):
                return tree[key][query[key]]
            else:
                return predict(query,tree[key][query[key]])
        
###################

###################
def DT(now,total,features,target_attribute_name="survived",parent_node_class = None):
	t1=target_attribute_name
	p1=parent_node_class
	if len(now)==0:
		return np.unique(total[t1])[np.argmax(np.unique(total[t1],return_counts=True)[1])]
	elif len(np.unique(now[t1])) < 2:
		return np.unique(now[t1])[0]
	elif len(features) ==0:
		return p1
	else:
		p1 = np.unique(now[t1])[np.argmax(np.unique(now[t1],return_counts=True)[1])]
		item_val = [Gain(now,feature,t1) for feature in features] 
		index= np.argmax(item_val)
		best_feat = features[index]
		tree = {best_feat:{}}
		features = [i for i in features if i != best_feat]
		for value in np.unique(now[best_feat]):
			value = value
			sub_data = now.where(now[best_feat] == value).dropna()
			subtree = DT(sub_data,total,features,t1,p1)
			if(subtree != 'survived'):
				tree[best_feat][value] = subtree
		return(tree)    
                
###################

###################

###################
def print_tree(tree):

	pprint(tree)
	return

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


#################we#        

################
name1=['pclass','age','gender']
tree = DT(dataset,dataset,name1,'survived',dataset.columns[:-1])

#use either of these print formats

#pretty(tree)
print_tree(tree)






# uncomment below line to test on queries
#test(dataset,tree)
#print("example run on few tests:")
#print('example test (crew,female,adult) : '+predict({'pclass':'crew','gender':'female','age':'adult'},tree,1))
#print('example test (1st,male,child) : '+predict({'pclass':'1st','gender':'male','age':'child'},tree,1))
#print('example test (crew,female,child) : ',end='')
thia = predict({'pclass':'2nd','gender':'female','age':'child'},tree,1)
print(thia)

#predict(query,tree,default = 1):
