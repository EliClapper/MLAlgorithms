import numpy as np
import pandas as pd

#Linear Discriminant Analysis for K groups. This example uses 3 groups, but any number can be used.

# generate data from normal dist
n = 100                 # sample size
np.random.seed(6164900) # set seed for random sampling
group1 = np.random.normal(loc = -2, scale = 1, size= n)    # generate n values from normal with mean 0.5
group2 = np.random.normal(loc = 0, scale = 1, size = n)  # generate n values from normal with mean -0.5
group3 = np.random.normal(loc = 2, scale = 1, size = n)  # generate n values from normal with mean -0.5
X = pd.Series(np.concatenate([group1, group2, group3]))       # append values to one pd series
y = pd.Series(np.repeat([0,1,2] , repeats = n))         # create groups by repeating each element in list [0,1] 100 times
df = pd.concat([y,X], axis = 1, keys= ['group', 'X']) # concatenate the two series to a dataframe


# to get the discriminant function of x, we need the group means, probabilities and weighted variance
def estimates(dataframe):
    levels = dataframe['group'].unique() # unqiue levels in group
    K = len(levels)               # how many groups in total 
    GroupList = [0]*K             # initalize list of length K
    for i in levels:              # Put X values of all K groups in GruopList
        GroupList[i] = dataframe[dataframe['group'] == i]
    N = sum([len(group) for group in GroupList])    # obtain total sample size
    Probs = [(len(group)/N) for group in GroupList] # obtain probabilities of being in a group
    Means = [group.mean().tolist()[1] for group in GroupList] # obtain mean for all groups
    Diffsqs = [((group - mean)**2).sum().tolist()[1] for group, mean in zip(GroupList, Means)] # obtain squared difference sum in all groups
    WeightedVar = sum(Diffsqs) / (N-K) # obtain weighted variance
    return([Means, WeightedVar, Probs]) # return list of metrics we need

#discrimination function
def deltax(value, dataframe):
    ests = estimates(dataframe)     # obtain estimates
    dx = [(value * (mean / ests[1]) - (mean**2 / (2*ests[1])) + np.log(prob)) for mean, prob in zip(ests[0], ests[2])]
    return(dx) # return the discrimination estimates

# function to loop over df
def simple_LDA(dataframe):
    preds = []                   # initiate memory
    for value in dataframe['X']: # loop over all values in the dataframe
        disc_vals = deltax(value, dataframe) # get discrimination values
        group = np.argmax(disc_vals) #get index of where disc val is largest
        preds.append(group) #append that index to predicted values
    return(preds)

# obtain predicted values for our dataframe and see the succesrate
predicted = pd.Series(simple_LDA(df))
SuccesRate = (predicted == df['group']).sum() / len(predicted) * 100
print(SuccesRate) # 86.0% predicted correctly


