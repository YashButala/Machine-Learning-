import pandas   as pd
import numpy as  np

def Laplacian_smooth(p1,p2):
	return (float(p1+1)/(p2+5))

def find_priori(data,class_num,col,val):
	if(class_num==1):
		c1=data['D'][data['D'] == 1].count()
	else:
		c1=data['D'][data['D'] == 0].count()
	true_match = data[data['D'] == class_num][[col]][data[data['D'] == class_num][[col]]==val].count()
	return Laplacian_smooth(true_match,c1)

def prob_matrix(data):
	return [[[find_priori(data, class_, col, val) for val in range(1, 6)] for col in data.columns[1:]] for class_ in range(0,2)]	
	
def nb_classifier(data,features,P_no,P_yes,P_matrix):
	zero_tmp=P_no
	one_tmp=P_yes
	r=len(features)-1
	for f in range(0,r):
		one_tmp = one_tmp* P_matrix[1][f][features[f+1]-1]

	for f in range(0,r):
		zero_tmp = zero_tmp* P_matrix[0][f][features[f+1]-1]
		
	if zero_tmp < one_tmp:
		return 1
	else:
		return 0

def get_accuracy(data,P_matrix):
	counter=0
	n_yes = data['D'][data['D'] == 1].count()
	n_no = data['D'][data['D'] == 0].count()
	total = data['D'].count()
	P_yes = n_yes/total
	P_no = n_yes/total
	for i,values in data.iterrows():
		counter = counter+1 if(values.iloc[0] == nb_classifier(data,values,P_no,P_yes,P_matrix)) else counter 
	acc = ((counter*100))/len(data.index)
	return acc

if __name__ == '__main__':
	res = open('data2_19_new.csv', 'w') 
	f = open('data2_19.csv', "r")
	text = f.readlines()
	for string in text:
		s1 = string[1:-2]
		print(s1,file=res)	
	res.close()
	f.close()	
	fields=["D","X1","X2","X3","X4","X5","X6"]
	train_data = pd.read_csv("data2_19_new.csv", header=1,names=fields)
	print("Training Data :") 
	print(train_data)
	P_matrix = prob_matrix(train_data)

	#print(P_matrix)
	res = open('test2_19_new.csv', 'w') 
	f = open('test2_19.csv', "r")
	text = f.readlines()
	for string in text:
		s1 = string[1:-2]
		print(s1,file=res)
	res.close()
	f.close()	
	fields=["D","X1","X2","X3","X4","X5","X6"]
	test_data = pd.read_csv("test2_19_new.csv", header=1,names=fields)
	print("Testing Data : ")
	print(test_data)
	print('Training Set Accuracy: ',get_accuracy(train_data,P_matrix))
	print('Test Set Accuracy: ',get_accuracy(test_data,P_matrix))
