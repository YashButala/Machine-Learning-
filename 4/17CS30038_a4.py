#####################################################################################
"""""
Roll Number - 17CS30038
Name - Yash Parag Butala
Assign - 4 (K Means Clustering)
numpy is used for array calculations
csv file handling is done using pandas 
this is a generalized code so can be used for other dataset
execution : python3 17CS30038_a4.py
"""
####################################################################################
import pandas   as pd
import numpy as  np
from pprint import  pprint
def dist(list1,list2 ):
	res=0.0
#	print(list1)
#	print(list2)
	for i in range(len(list1)):
		res+=(list1[i]-list2[i])*(list1[i]-list2[i])
	return res	
def intersect(a, b):
    """ return the intersection of two lists """
    return len(list(set(a) & set(b)))

def union(a, b):
    """ return the union of two lists """
    return len(list(set(a) | set(b)))    


dataset = pd.read_csv('data4_19.csv',header=None )
# print (dataset)
k = 3
np.random.seed(0)
means_index = np.random.randint(low=0, high=len(dataset.index)-1, size=k)
means=[]
for i in range (k):
	i1=means_index[i]
	tmp = [dataset.iloc[i1][0],dataset.iloc[i1][1],dataset.iloc[i1][2],dataset.iloc[0][3]]
	means.append(tmp)

# print(means_index[0])
# print( means[0][0])
# print( means[0][1])
#print( means[0][2])
#means_index[0][1]+" "+means[0][2])	
j1=[]
j2=[]
j3=[]
i1=[]
i2=[]
i3=[]
for i in range(len(dataset.index)):
	if dataset.iloc[i][4]=="Iris-setosa":
		i1.append(i)
	if dataset.iloc[i][4]=="Iris-versicolor":
		i2.append(i)	
	if dataset.iloc[i][4]=="Iris-virginica":
		i3.append(i)		
# print(i1)
# print(i2)
# print(i3)
for i in range(10):
	type1=[]
	type2=[]
	type3=[]
	for j in range(len(dataset.index)):
		curr = [dataset.iloc[j][0],dataset.iloc[j][1],dataset.iloc[j][2],dataset.iloc[j][3]]
		dist1 = dist( curr,means[0] )
		dist2 = dist( curr ,means[1]) 
		dist3 = dist(curr ,means[2])  
		if(dist1<dist2 and dist1<dist3):
			type1.append(curr)
			if i==9:
				j1.append(j)
		elif(dist2<dist1 and dist2<dist3):
			type2.append(curr)
			if i==9:
				j2.append(j)
		else:
			type3.append(curr)			
			if i==9:
				j3.append(j)
	means[0] = np.mean(type1, axis = 0)
	means[1] = np.mean(type2, axis = 0)
	means[2] = np.mean(type3, axis = 0)
	# print(means[0])
	# print(means[1])
	# print(means[2])	
# print(j1)
# print(j2)
# print(j3)

jacc_index1 = 1.0 - intersect(i1,j1)/union(i1,j1)
jacc_index2 = 1.0 - intersect(i1,j2)/union(i1,j2)
jacc_index3 = 1.0 - intersect(i1,j3)/union(i1,j3)
jacc_index4 = 1.0 - intersect(i2,j1)/union(i2,j1) 	 
jacc_index5 = 1.0 - intersect(i2,j2)/union(i2,j2)
jacc_index6 = 1.0 - intersect(i2,j3)/union(i2,j3)
jacc_index7 = 1.0 - intersect(i3,j1)/union(i3,j1)
jacc_index8 = 1.0 - intersect(i3,j2)/union(i3,j2)
jacc_index9 = 1.0 - intersect(i3,j3)/union(i3,j3)

print("The means of 3 clusters after 10 iterations are:")
print("")
print("Cluster 1:"," ",means[0][0]," ",means[0][1]," ",means[0][2]," ",means[0][3])
print("")
print("Cluster 2:"," ",means[1][0]," ",means[1][1]," ",means[1][2]," ",means[1][3])
print("")
print("Cluster 3:"," ",means[2][0]," ",means[2][1]," ",means[2][2]," ",means[2][3])



print("\n\n")


print("Jaccard dustance with Iris-setosa is:")
print('CLuster 1 : ',jacc_index1)
print('CLuster 2 : ',jacc_index2)
print('Cluster 3 : ',jacc_index3)
print("Jaccard dustance with Iris-versicolor is:")
print('CLuster 1 : ',jacc_index4)
print('CLuster 2 : ',jacc_index5)
print('Cluster 3 : ',jacc_index6)
print("Jaccard dustance with Iris-virginica is:")
print('CLuster 1 : ',jacc_index7)
print('CLuster 2 : ',jacc_index8)
print('Cluster 3 : ',jacc_index9)