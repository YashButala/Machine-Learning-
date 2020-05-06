#####################################################################################
"""""
Roll Number - 17CS30038
Name - Yash Parag Butala
Assign - 3 (Adaboost)
numpy is used for array calculations
csv file handling is done using pandas 
this is a generalized code so can be used for other dataset
execution : python3 17CS30038_a3.py
"""
####################################################################################
import pandas   as pd
import numpy as  np
from pprint import  pprint

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
if __name__ == '__main__':
    dataset = pd.read_csv('data3_19.csv', names=['pclass','age','gender','survived'])
    #print(dataset)
    name1=['pclass','age','gender']
    tree_size = int(len(dataset.index)*0.5)
    itr_count = 3
    classifier_weight_list = []
    weights = np.ones(len(dataset.index), dtype=float) / len(dataset.index)
    trees = []
    indices = np.arange(len(dataset.index))
    for i in range(itr_count):
        sub_data = []
        sample = np.random.choice(indices, tree_size, p = weights)
        for s in sample:
            sub_data.append(dataset.iloc[s])
        sample_df = pd.DataFrame(sub_data, columns=['pclass','age','gender','survived'])    
    # train the classifier
        tree = DT(sample_df,sample_df,name1,'survived',sample_df.columns[:-1])
    #    print(tree)
    # calculate weight of the classifier
        sample_targets = sub_data[-1]       
        count = 0     #count of correct prediction
        for s in sub_data:
  #          print(s[0]+" "+s[1]+" "+s[2])
            query = {'pclass': s[0],'age':s[1],'gender':s[2]}
           # print(query)
            prediction = predict(query,tree,1)
     #       print(prediction )
      #      print(s[len(s)-1])
      #increase count if prediction matches true val
            if prediction == s[len(s)-1]:       
                count +=  1       
                    # print(count)
            # print(count)
            # print('Local Accuracy:', count * 100.0 /len(sub_data))  #prints accuracy

            
        e = ((len(sub_data)-count) * 1.0) /len(sub_data)
     #   print(count)
     #   print(e)

        weight = np.log((1 - e) / (e + 1e-6))
        weight = weight * 0.5
        classifier_weight_list.append(weight)

    # update the weights
        s_count=0
        for s in sub_data:
     #  y_pred = predict_label(tree, attr_dict, data, s)
            query = {'pclass': s[0],'age':s[1],'gender':s[2]}
            y_pred = predict(query,tree,1)
            if y_pred == s.iloc[-1]:
                exp_weight = np.exp(-1.0*weight)
            else:
                exp_weight = np.exp(weight)
            weights[s_count] = weights[s_count]*np.exp(exp_weight)
            s_count = s_count+1
        sum_weights = np.sum(weights)
        weights = weights / sum_weights

    # save the classifier
        trees.append(tree)

    # load the test set
    testset = pd.read_csv('test3_19.csv', names=['pclass','age','gender','survived'])
 #   print(testset)
    sample_test=[]
    for s in range(len(testset.index)):
        sample_test.append(testset.iloc[s])
    pos_count = 0
    # predict on the test set
    for s in sample_test:
        pred = 0
        query = {'pclass': s[0],'age':s[1],'gender':s[2]}    
        for i in range(itr_count):
            y_pred = predict(query,tree,1)
            if y_pred == s.iloc[-1]:
                classifier_weight = classifier_weight_list[i]
            else:
                classifier_weight = -classifier_weight_list[i]
            pred+=classifier_weight
        if pred >= 0:
            pos_count += 1

    print('Test Accuracy on given data: {} percent'.format((100.0*pos_count) / len(testset.index)))

