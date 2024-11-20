#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
np.set_printoptions(precision=5, suppress=True, linewidth=160)

filename1 = 'housing.csv'
data = pd.read_csv(filename1, sep=',',header=None)
n_cols = 5
n_train = 253
n_test  = 253

#print(data.head())
def r_squared(actual, fit_vals):
    """
    Computes the r^2 value for values calculated from the least squares fit 
    
    In: actual - the true values of the homes, fit_vals - the values calculated using the regression
    coefficients from the least squares fit
    Returns: rsquared - the r^2 value of this fit
    """
    RSS = np.sum((actual - fit_vals)**2) #sum of squares of residuals
    TSS = np.sum((actual - np.mean(actual))**2) #total sum of squares
    r2 = 1 - (RSS / TSS) #rsquared value
    return r2

B = np.zeros((n_train,n_cols))
B1 = np.zeros((n_train,n_cols))
b = np.zeros((n_train,1))
C = np.zeros((n_test,n_cols))
C1 = np.zeros((n_test,n_cols))
c = np.zeros((n_test,1))

### Training Set ###
B[0:n_train,0:1]  = data.iloc[0:n_train,5:6]    # RM rooms per dwelling
B[0:n_train,1:2]  = data.iloc[0:n_train,10:11]  # PTRATIO parent teacher ratio
B[0:n_train,2:3]  = data.iloc[0:n_train,12:13]  # LSTAT % lower status of population
B[0:n_train,3:4]  = data.iloc[0:n_train,7:8]    # DIS distance to Boston employment centres
B[0:n_train,4:5]  = data.iloc[0:n_train,9:10]   # TAX property tax per $10,000

b[0:n_train,0:1]  = data.iloc[0:n_train,13:14] #this is the median value of home in 1000s of dollars

### Testing Set ###
C[0:n_test,0:1]  = data.iloc[n_train:n_train+n_test,5:6]    # RM rooms per dwelling
C[0:n_test,1:2]  = data.iloc[n_train:n_train+n_test,10:11]  # PTRATIO parent teacher ratio
C[0:n_test,2:3]  = data.iloc[n_train:n_train+n_test,12:13]  # LSTAT % lower status of population
C[0:n_test,3:4]  = data.iloc[n_train:n_train+n_test,7:8]    # DIS distance to Boston employment centres
C[0:n_test,4:5]  = data.iloc[n_train:n_train+n_test,9:10]   # TAX property tax per $10,000

c[0:n_test,0:1]  = data.iloc[n_train:n_train+n_test,13:14] #the median value of home in 1000s of dollars


#############
#### 1 A ####   
#############


##least squares: 
x = np.linalg.lstsq(B,b) #x[0] is soln, 1 is residuals, 2 is rank, 3 is singular values 
##This is the solution to A x = b 
##use x to find b' in A' x = b'

b_prime1 = np.dot(C,x[0]) #bprime1 is c, but calculated using the B regression values
b_prime2 = np.dot(B,x[0]) #bprime2 is b, calculated using the B regression values
#b_primes are calculated from the least square regression from the training set

#R squared values: 
r2_1 = r_squared(c,b_prime1) #R^2 of test set with values calculated from training set regression
r2_2 = r_squared(b,b_prime2) #R^2 of training set with values calculated from training set regression
print('Q1 a)')
print('Coefficients found from training set:')
print(x[0])

print('R Squared values from comparing the predictions made by the training set regression to:')
print(f'1. the test set: {r2_1}')
print(f'2. the training set: {r2_2}')
print('')
### Now to use the test set to do the same for the training set ###
x = np.linalg.lstsq(C,c)
c_prime1 = np.dot(B,x[0]) 
c_prime2 = np.dot(C,x[0])
#c_primes are calculated from the least squares regression from the test set. 

#R squared values: 
r2_3 = r_squared(c,c_prime2) #R^2 of test set with values calculated from test set regression
r2_4 = r_squared(b,c_prime1) #R^2 of training set with values calculated from test set regression

print('Coefficients found from test set:')
print(x[0])
print('')
print('R Squared values from comparing the predictions made by the test set regression to:')
print(f'1. the test set: {r2_3}')
print(f'2. the training set: {r2_4}')
print('')


#############
#### 1 B ####   
#############


print('Q1 b)')
fig, ax = plt.subplots(dpi=500)

ax.plot(c,b_prime1,'.', c = 'tab:orange', label = 'Calculated from training coefficients')
ax.plot(c,c_prime2,'.', c = 'tab:blue', label = 'Calculated from test coefficients')
ax.set_xlabel('Test Set Actual Median Value (/1000$)')
ax.set_ylabel('Calculated Median Value (/1000$)')
ax.legend(loc='upper left')
ax.text(35,0,f'R Squared = {r2_1:.4g}', c = 'tab:orange')
ax.text(35,-2.5,f'R Squared = {r2_3:.4g}', c = 'tab:blue')
plt.show()

fig, ax = plt.subplots(dpi=500)
ax.plot(b,c_prime1,'.', c = 'tab:blue', label = 'Calculated from test coefficients')
ax.plot(b,b_prime2,'.', c = 'tab:orange', label = 'Calculated from training coefficients')
ax.set_xlabel('Training Set Actual Median Value (/1000$)')
ax.set_ylabel('Calculated Median Value (/1000$)')
ax.text(45,10,f'R Squared = {r2_2:.4g}', c = 'tab:orange', ha= 'center')
ax.text(45,7.5,f'R Squared = {r2_4:.4g}', c = 'tab:blue', ha= 'center')
ax.legend(loc='upper left')



plt.show()


#############
#### 1 C ####   
#############


print('Q1 c)')
#transpose B and C
B = B.transpose()
C = C.transpose()

def correlation_matrix(M, n_cols,length):
    """
    Finds the correlation matrix of a matrix 
    In: M - matrix of values, n_cols - number of columns, length - lenght of matrix
    Returns: correlation matrix, covariance matrix, the centered matrix, the centered matrix transposed
    """
    #M = M.transpose() #just added the transpose 
    M2 = np.zeros((n_cols, length))
    #center the data - this ensures that the covaraince is about the mean of the data 
    #do z scores instead? 
    means = [np.sum(M[i])/len(M[i]) for i in range((n_cols))]
    for i in range((n_cols)):
        M2[i] = M[i] - means[i]
    #transpose matrix
    M2_T = M2.transpose()
    
    covariance = np.dot(M2,M2_T)/(length-1)
    std_dev = np.sqrt(np.diag(covariance)) #stdeviation is the sqrt of the diagonals of the cov matrix
    correlation = covariance / np.outer(std_dev, std_dev) #outer product gets sigma_i * sigma_j
    
    return correlation, covariance, M2, M2_T

corrB = correlation_matrix(B,n_cols,n_train)[0] #Correlation matrix for Training set
corrC = correlation_matrix(C,n_cols,n_test)[0] ##Correlation matrix for Test set
print('Correlation matrix for triaining set:')
print(corrB)
print('')
print('Correlation matrix for test set:')
print(corrC)


#############
#### 1 D ####   
#############


print('Q1 d)')
#plotting variables with high and low correlation: 
fig,ax = plt.subplots(dpi=700)
#2nd and 4th:
#0.01 to 0.36
###
ax.scatter( B[3]/np.sum(B[3]), B[1]/np.sum(B[1]) ,s=7, c = 'tab:orange') 
ax.scatter(C[3]/np.sum(C[3]), C[1]/np.sum(C[1]),s=7, c = 'tab:orange', label ='Distances to employment - Parent teacher ratio')
#3rd and 4th
#-0.34 to -0.57
ax.scatter( B[3]/np.sum(B[3]), B[2]/np.sum(B[2]) ,s=7, c = 'tab:red') 
ax.scatter(C[3]/np.sum(C[3]), C[2]/np.sum(C[2]),s=7, c = 'tab:red', label='Distances to employment - % lower status of population')

#-0.71 to 0.54
ax.scatter( B[0]/np.sum(B[0]), B[2]/np.sum(B[2]) ,s=7, c = 'tab:blue') #1st and 3rd variable: rooms per dwelling vs % lower status of population 
ax.scatter(C[0]/np.sum(C[0]), C[2]/np.sum(C[2]),s=7, c = 'tab:blue', label = 'Number of Rooms - % lower status of population')

ax.set_xlabel('Relative Value of labelled data')
ax.legend(loc='upper right')
ax.set_ylabel('Relative Value of labelled data')
ax.set_xlim(0,0.01)
plt.show()


#############
#### 1 E ####   
#############


print('Q1 e)')
def principal_components(corr, n_PC):
    """"
    Finds the prinicipal component vectors from eigenvectors of correlation matrix 
    In: corr - correlation matrix, n_PC - number of principal components to find
    Returns: Principal component vectors, arguments to show which is which 
    """
    e_vals, e_vects = np.linalg.eig(corr)
    #find the arguments required to put into descending order
    arguments_descending = np.argsort(e_vals)[::-1]
    #rearrange eigenvectors and values
    e_vals = e_vals[arguments_descending]
    e_vects = e_vects[:, arguments_descending]
    
    principal_vects = e_vects[:, :n_PC] #take the first n_PC eigenvectors as principals
    principal_vects = principal_vects.transpose()
    
    return principal_vects, arguments_descending

PC_vects, args = principal_components(corrB, 2)

print(args)
#these are found to be the 2st and the 5th, Room numbers and tax 


_,_, B_new, B_new_T = correlation_matrix(B,n_cols,n_train) 
_,_, C_new, C_new_T = correlation_matrix(C,n_cols,n_test) 

#project the training set onto the eigenvectors:
proj1 = np.dot(PC_vects[0],B_new)
proj2 = np.dot(PC_vects[1],B_new)

fig,ax = plt.subplots(dpi =500)
ax.scatter(proj1,proj2,s=15,label = 'Training Set Projection')

#same for test set 
proj1 = np.dot(PC_vects[0],C_new)
proj2 = np.dot(PC_vects[1],C_new)
ax.scatter(proj1,proj2, s =15,label = 'Test Set Projection')
ax.set_ylabel('Principle Component 2')
ax.set_xlabel('Principle Component 1')
ax.legend(loc='upper left')

plt.show()
#lovely straight line! - captures it well

##Also can be done for test set but not asked for. 
#PC_vects2, args2 = principal_components(corrC, 2)
#proj1 = np.dot(PC_vects2[0],B_new)
#proj2 = np.dot(PC_vects2[1],B_new)
#fig,ax = plt.subplots()
#ax.scatter(proj1,proj2)
#proj1 = np.dot(PC_vects2[0],C_new)
#proj2 = np.dot(PC_vects2[1],C_new)
#ax.scatter(proj1,proj2)

#### R^2 values ####
projection1 = np.dot(B_new_T, PC_vects.transpose())
projection2 = np.dot(C_new_T, PC_vects.transpose())

#new R^2 as it requires going along difference axes. RSS/TSS is same as sum of variance
r2_projB = 1 - np.sum(np.var(projection1, axis=0))/ np.sum(np.var(B,axis=1))
r2_projC = 1 - np.sum(np.var(projection2, axis=0))/ np.sum(np.var(C,axis=1))
print("R squared values for the projection of the training set and test set projected onto the 2 principal component vectors")
print(f'Training set R^2: {r2_projB}')
print(f'Test setR^2: {r2_projC}')


