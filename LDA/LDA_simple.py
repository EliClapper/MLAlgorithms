import numpy as np
import pandas as pd

# NOTE, algorithm not yet generalized to variable K and p.

# generate data from normal dist
n = 100                 # sample size
np.random.seed(6164900) # set seed for random sampling
group1 = np.random.normal(loc = 1, scale = 1, size= n)    # generate n values from normal with mean 0.5
group2 = np.random.normal(loc = -1, scale = 1, size = n)  # generate n values from normal with mean -0.5
X = pd.Series(np.concatenate([group1, group2]))       # append values to one pd series
y = pd.Series(np.repeat([0,1] , repeats = n))         # create groups by repeating each element in list [0,1] 100 times
df = pd.concat([y,X], axis = 1, keys= ['group', 'X']) # concatenate the two series to a dataframe

# to get the discriminant function of x, we need the group means, probabilities and weighted variance
def estimates(dataframe):
    K = dataframe['group'].nunique()                  # number of groups is unique elements in group
    group1 = dataframe[dataframe['group'] == 0]['X']  # obtain series for group 1
    group2 = dataframe[dataframe['group'] == 1]['X']  # series with values for group 2
    N =  len(group1) + len(group2)                    # total sample size
    probs = pd.Series([len(group1) / N, len(group2) / N]) # obtain probability of being in a group
    means = pd.Series([group1.mean(), group2.mean()])     # obtain series of group means

    group1_diffsq = ((group1 - means[0])**2).sum()        # get squared differences for group 1
    group2_diffsq = ((group2 - means[1])**2).sum()        # for group2

    weighted_Var = (group1_diffsq + group2_diffsq) / (N-K) # obtain weighted variance

    return([means, weighted_Var, probs])                    # return list of lists


#discrimination function
def deltax(value, dataframe):
    ests = estimates(dataframe)     # obtain estimates
    dx = value * (ests[0] / ests[1] - (ests[0]**2)/ (2*ests[1])) + np.log(ests[2]) # obtain discrimination estimate for both groups
    return(dx) # return the discrimination estimates

# function to loop over df
def simple_LDA(dataframe):
    preds = []                   # initiate memory
    for value in dataframe['X']: # loop over all values in the dataframe
        disc_vals = deltax(value, dataframe) # get discrimination values
        group = int(disc_vals[disc_vals == max(disc_vals)].index.values) #get index of where disc val is largest
        preds.append(group) #append that index to predicted values
    return(preds)

# obtain predicted values for our dataframe and see the succesrate
predicted = pd.Series(simple_LDA(df))
SuccesRate = (predicted == df['group']).sum() / len(predicted) * 100
print(SuccesRate) # 86.0% predicted correctly


