################################################################################
#NAME: Baseball_Forward_final.py
#
#CREATED BY: Ben Mooneyham
#
#DESCRIPTION: This script uses Gaussian Mixture Model clustering algorithms to 
#identify prototypical "types" of MLB Batters and Pitchers.It then uses this 
#"type" classification data within a Support Vector Machine supervised learning 
#algorithm to predict MLB team playoff berths from teams' compositions by player
#"type."
################################################################################


#Import Necessary Modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn import svm
import matplotlib.cm as cm
import itertools
from numpy.ma import masked_array


#Set Working Directory
os.chdir("/Users/benmooneyham/Desktop/Baseball/")
workdir = os.getcwd()


#Declare Global Variables
global Pitching, Batting, g, h, BandP_trim, Team_Success_trim, X, Y 
global n_components_B, n_components_P


#General Settings
n_teams = 30 #Number of MLB teams
test_yrs = 1 #Number of seasons back in time (from 2015) to use in test dataset
rs = 8 #Random seed for reproducibility


#Import Data
#Batting statistics
Batting = pd.read_csv("batting100.csv")
Batting.drop('SBp', axis=1, inplace=True) #Remove SBp
Batting_Categories = Batting.columns.values.tolist() #Get batting categories
X=Batting.iloc[:,5:28] #Select variables for use in analysis
#Pitching statistics
Pitching = pd.read_csv("pitching50.csv")
Pitching_Categories = Pitching.columns.values.tolist() #Get pitching categories
Y=Pitching.iloc[:,4:15] #Select variables for use in analysis
Y = Y[np.isfinite(Y['BAOpp'])] #Remove NaNs
#Team statistics
Teams = pd.read_csv("teams.csv")
Teams_Categories = Teams.columns.values.tolist() #Get team category names
Team_Success1 = Teams['WCWin'] #Code for Wild Card winners
Team_Success2 = Teams['DivWin'] #Code for Division winners
Team_Success = np.empty([len(Team_Success1)])
#Code whether teams made the playoffs (i.e., either won divison or Wild Card)
for entry in range(len(Team_Success1)):
    if Team_Success1[entry] == 'Y' or Team_Success2[entry] == 'Y':
        Team_Success[entry] = 1
    else:
        Team_Success[entry] = 0


#Determine Best Number of Clusters for Batting and Pitching Data Separately
lowest_aic_B = np.infty
lowest_aic_P = np.infty
aic_B = []
aic_P = []
n_components_range = range(8, 12) #Can edit for speed
cv_types = ['full','spherical', 'tied', 'diag', 'full'] #Can edit for speed
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a mixture of Gaussians with Expectation-Maximization algorithm
        gmm_B = mixture.GMM(n_components=n_components, covariance_type=cv_type, 
        random_state = rs) #Batting    
        gmm_B.fit(X)
        gmm_P = mixture.GMM(n_components=n_components, covariance_type=cv_type, 
        random_state = rs) #Pitching 
        gmm_P.fit(Y)
        aic_B.append(gmm_B.aic(X))
        aic_P.append(gmm_P.aic(Y))
        #Record best AIC
        if aic_B[-1] < lowest_aic_B:
            lowest_aic_B = aic_B[-1]
            best_gmm_B = gmm_B
        if aic_P[-1] < lowest_aic_P:
            lowest_aic_P = aic_P[-1]
            best_gmm_P = gmm_P
aic_B = np.array(aic_B)
aic_P = np.array(aic_P)


#Plot the AIC scores (if desired)
color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
bars = []
bars_P = []
fig1,ax1 = plt.subplots(2, figsize=(10,8))
#Batting AICs
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(ax1[0].bar(xpos, aic_B[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
ax1[0].set_xticks(n_components_range)
ax1[0].set_ylim([aic_B.min() * 1.01 - .01 * aic_B.max(), aic_B.max()])
ax1[0].set_title('AIC score per model - Batting')
xpos = np.mod(aic_B.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(aic_B.argmin() / len(n_components_range))
ax1[0].text(xpos, aic_B.min() * 0.97 + .03 * aic_B.max(), '*', fontsize=14)
ax1[0].set_xlabel('Number of components')
ax1[0].legend([b[0] for b in bars], cv_types, loc=0)
#Pitching AICs
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars_P.append(ax1[1].bar(xpos, aic_P[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
ax1[1].set_xticks(n_components_range)
ax1[1].set_ylim([aic_P.min() * 1.01 - .01 * aic_P.max(), aic_P.max()])
ax1[1].set_title('AIC score per model - Pitching')
xpos = np.mod(aic_P.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(aic_P.argmin() / len(n_components_range))
ax1[1].text(xpos, aic_P.min() * 0.97 + .03 * aic_P.max(), '*', fontsize=14)
ax1[1].set_xlabel('Number of components')
ax1[1].legend([b[0] for b in bars_P], cv_types, loc=0)


#Set Number of Clusters and Covariance Type to Use for GMM (Based on AIC)
#Batting
n_components_B = best_gmm_B.n_components #Use number of clusters based on lowest AIC score for Batting data
g = mixture.GMM(n_components_B,covariance_type=best_gmm_B.covariance_type, random_state = rs) #Run GMM
X_results = g.fit_predict(X) #Assign cluster labels to individual batters
np.set_printoptions(suppress=True, precision=4)
B = np.round(g.means_, 2) #Record cluster centers for visualization
#Pitching
n_components_P = best_gmm_P.n_components #Use number of clusters based on lowest AIC score for Pitching data
h = mixture.GMM(n_components_P,covariance_type=best_gmm_P.covariance_type, random_state = rs) #Run GMM
Y_results = h.fit_predict(Y) #Assign cluster labels to individual pitchers
np.set_printoptions(suppress=True, precision=4)
P = np.round(h.means_, 2) #Record cluster centers for visualization


#Plot Cluster Center Values
#Batting
figB,axB = plt.subplots(figsize=(15,15))
for i in range(len(B.T)):
    c1 = masked_array(B, mask=(np.ones_like(B)*(B[0]!=B[0][i-1]))) 
    axB.matshow(c1,cmap=cm.Blues)
    axB.text(i,(len(B)), Batting_Categories[5+i], fontsize=12, rotation =30, horizontalalignment='center') 
for i in range(len(B)):    
    axB.text(-2.5,(i-.1), 'Type %s' % (i+1), fontsize=12, horizontalalignment='left') 
axB.set_ylim([-0.5,n_components_B-0.5])
axB.set_xticks([])
axB.set_yticks([])
axB.plot([14.5, 14.5], [-0.5, (n_components_B-0.5)], color='k', linestyle='--', linewidth=2)
#Pitching
figP,axP = plt.subplots(figsize=(15,15))
for j in range(len(P.T)):
    c2 = masked_array(P, mask=(np.ones_like(P)*(P[0]!=P[0][j-1]))) 
    axP.matshow(c2,cmap=cm.Greens)
    axP.text(j,(len(P)-0.25), Pitching_Categories[4+j], fontsize=12, rotation =30, horizontalalignment='center', verticalalignment='bottom') 
for j in range(len(P)):    
    axP.text(-2.5,(j-.1), 'Type %s' % (j+1), fontsize=12, horizontalalignment='right')
axP.set_ylim([-0.5,n_components_P-0.5])
axP.set_xticks([])
axP.set_yticks([])
axP.plot([6.5, 6.5], [-0.5, (n_components_P-0.5)], color='k', linestyle='--', linewidth=2)


#Tabulate Clusters by Year and Team
#Add cluster labels to batting and pitching data
Batting.loc[:,'cluster'] = pd.Series(X_results, index=Batting.index)
Pitching = Pitching[np.isfinite(Pitching['BAOpp'])]
Pitching.loc[:,'cluster'] = pd.Series(Y_results, index=Pitching.index)
#Create Cluster-wise dummy variables
#Batting
dummies_B = pd.get_dummies(Batting['cluster'])
col_names_dummies_B = dummies_B.columns.values
for i,value in enumerate(col_names_dummies_B):
        Batting[value] = dummies_B.iloc[:,i]
Batting_counts = Batting.groupby(['yearID','teamID']).sum()
Batting_counts.drop(Batting_Categories[4:], axis=1, inplace=True)
Batting_counts.drop('cluster', axis=1, inplace=True)
#Pitching
dummies_P = pd.get_dummies(Pitching['cluster'])
col_names_dummies_P = dummies_P.columns.values
for i,value in enumerate(col_names_dummies_P):
        Pitching[value] = dummies_P.iloc[:,i]
Pitching_counts = Pitching.groupby(['yearID','teamID']).sum()
Pitching_counts.drop(Pitching_Categories[4:], axis=1, inplace=True)
Pitching_counts.drop('cluster', axis=1, inplace=True)


#Make list of inputs for SVM
BandP = pd.concat([Batting_counts, Pitching_counts], axis=1) #Combine batting and pitching cluster counts
Missing = ~np.isnan(BandP).any(axis=1)
Missing_array = np.array(Missing)
BandP_trim = BandP[Missing]
Team_Success_trim = Team_Success[Missing_array]


#Run SVM 
Classify = svm.SVC(class_weight='balanced', random_state = rs) #Set SVM parameters
Classify.fit(BandP_trim[:-(n_teams*test_yrs)][:], Team_Success_trim[:-(n_teams*test_yrs)][:]) #Fit SVM model to training data (i.e., all data prior to test data)
Training_Accuracy = Classify.score(BandP_trim[:-(n_teams*test_yrs)][:], Team_Success_trim[:-(n_teams*test_yrs)][:]) #Calculate accuracy of SVM model within training dataset
Predictions = Classify.predict(BandP_trim[-(n_teams*(test_yrs+1)):-(n_teams)][:]) #Make predictions for test dataset
Prediction_Comparison = [Predictions, Team_Success_trim[-(n_teams*test_yrs):][:]]


#Code for Hits, False Positives, Misses, and Correct Rejections
Prediction_Hits = np.empty([len(Prediction_Comparison[0])])
Prediction_FPs = np.empty([len(Prediction_Comparison[0])])
Prediction_Misses = np.empty([len(Prediction_Comparison[0])])
Prediction_CRs = np.empty([len(Prediction_Comparison[0])])
for entry in range(len(Prediction_Comparison[0])):
    #Hits
    if Prediction_Comparison[0][entry] == 1  and Prediction_Comparison[1][entry] == 1:
        Prediction_Hits[entry] = 1
    else:
        Prediction_Hits[entry] = 0
    #False Positives
    if Prediction_Comparison[0][entry] == 1  and Prediction_Comparison[1][entry] == 0:
        Prediction_FPs[entry] = 1
    else:
        Prediction_FPs[entry] = 0
    #Misses
    if Prediction_Comparison[0][entry] == 0  and Prediction_Comparison[1][entry] == 1:
        Prediction_Misses[entry] = 1
    else:
        Prediction_Misses[entry] = 0
    #Correct Rejections
    if Prediction_Comparison[0][entry] == 0  and Prediction_Comparison[1][entry] == 0:
        Prediction_CRs[entry] = 1
    else:
        Prediction_CRs[entry] = 0


#Calculate Measures of SVM Model Performance 
Prediction_Accuracy = 1-sum(abs(Prediction_Comparison[0] - Prediction_Comparison[1]))/len(Prediction_Comparison[0])
Sensitivity = sum(Prediction_Hits)/(sum(Prediction_Hits)+sum(Prediction_Misses))
Specificity = sum(Prediction_CRs)/(sum(Prediction_CRs)+sum(Prediction_FPs))
PPV = sum(Prediction_Hits)/(sum(Prediction_Hits)+sum(Prediction_FPs)) #Positive Predictive Value
NPV = sum(Prediction_CRs)/(sum(Prediction_CRs)+sum(Prediction_Misses)) #Negative Predictive Value


#Print SVM Model Outputs
print 'Training_Accuracy = %04.2f' % Training_Accuracy
print 'Prediction_Accuracy = %04.2f' % Prediction_Accuracy
print 'Sensitivity = %04.2f' % Sensitivity
print 'Specificity = %04.2f' % Specificity
print 'PPV = %04.2f' % PPV
print 'NPV = %04.2f' % NPV


##Show Figures
#plt.show()
#plt.tight_layout()


##Save Figures
#figB.savefig('Batting_Clusters_final.png', dpi=400, facecolor='w', edgecolor='k',
#       transparent=False, bbox_inches='tight', pad_inches=0.1) 
#figP.savefig('Pitching_Clusters_final.png', dpi=400, facecolor='w', edgecolor='k',
#       transparent=False, bbox_inches='tight', pad_inches=0.1) 