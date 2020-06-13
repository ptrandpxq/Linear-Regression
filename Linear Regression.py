import pandas
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


dataset = pandas.read_csv('Salary_Data.csv')
X = dataset.iloc[:,0].values 
Y = dataset.iloc[:,-1].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 0)
X_train = X_train.reshape(-1,1)
Y_train = Y_train.reshape(-1,1)

theta0 = 0
theta1 = 0


iteration = 5000

a = 0.02
x = np.linspace(0, 12,30)
#print(x)
y = theta0+theta1*x



for i in range(iteration):
    c1=0
    c2=0
    for j in range(len(X_train)):

	    temp0 = float((theta0+theta1*X_train[j]-Y_train[j]))
	    #print('temp',temp)
	    c1 += temp0
	    #print('c1:',c1)
	    c2 += float((theta0+theta1*X_train[j]-Y_train[j])*X_train[j])
	    #print('loss:',y[j]-Y_train[j])



    print('iteration: ',i,''c1)
    if c1 < 0.0001 and c1> -0.0001:
    	break



    #print('--------------------------------------------------------')
   
    theta0 = theta0 - a*(1/len(X_train))*c1
    theta1 = theta1 - a*(1/len(X_train))*c2

  

#print('c2:',c2)

print(theta0)
print(theta1)



plt.figure(figsize=(8,6)) #设置画布大小

plt.scatter(X_train, Y_train,color = 'red')  #描出训练集对应点matplotlib
x = np.linspace(0, 12,1000)
y = theta0+theta1*x
plt.plot(x,y)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Train set')

plt.show()
