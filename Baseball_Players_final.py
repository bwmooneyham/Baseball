################################################################################
#NAME: Baseball_Players_final.py
#
#CREATED BY: Ben Mooneyham
#
#DESCRIPTION: This script uses variables produced by "Baseball_Forward_final.py" 
# to a) identify the batters and pitcher who most nearly match the prototypical 
# "types" identified by the Gaussian Mixture Model clustering algorithms, and b)
# calculate the mean number of players of each "type" on MLB playoff teams. 
################################################################################


#Import Necessary Modules
import numpy as np
import matplotlib.pyplot as plt


#Import Global Variables
from Baseball_Forward_final import (Pitching, Batting, g, h, BandP_trim, 
    Team_Success_trim, X, Y, n_components_B, n_components_P)


#Limit Player Identification to 2015 only (so they are recognizable/current)
Pitching = Pitching.reset_index(drop=True)
Batting_2015 = Batting.loc[Batting['yearID'] == 2015]
Pitching_2015 = Pitching.loc[Pitching['yearID'] == 2015]
Batting_2015 = Batting_2015.reset_index(drop=True)
Pitching_2015 = Pitching_2015.reset_index(drop=True)


#Determine Prototypical Players
#Calculate player fits to cluster assignments
Player_fits_B = g.predict_proba(X[5999:][:]) #Batting
Player_fits_P = h.predict_proba(Y[3433:][:]) #Pitching
#Record indices of best fitting players
BestB_index = np.argmax(Player_fits_B[:][:], axis=0) #Batting
BestP_index = np.argmax(Player_fits_P[:][:], axis=0) #Pitching
#Select relevant variables
Batter = Batting_2015['playerID']
Batter_Team =Batting_2015['teamID']
Pitcher = Pitching_2015['playerID']
Pitcher_Team =Pitching_2015['teamID']
#Identify best-fitting players and their team
Batter[BestB_index]
Batter_Team[BestB_index]
Pitcher[BestP_index]
Pitcher_Team[BestP_index]


#Calculate Mean Number of Players of Each Type on Each Playoff Team
SuccessData = BandP_trim.loc[Team_Success_trim ==1] #Playoffs only
#Separate batting and pitching cluster counts
SuccessDataB = SuccessData.T[:-n_components_P][:]
SuccessDataP = SuccessData.T[(n_components_B):][:]
#Calculate means
meansB = np.empty([len(SuccessDataB)])
meansP = np.empty([len(SuccessDataP)])
SuccessDataB_column_names = SuccessDataB.T.columns #Batting
for i, value in enumerate(SuccessDataB_column_names.values):
    meansB[value] = np.mean(SuccessDataB.T.iloc[:,i])
SuccessDataP_column_names = SuccessDataP.T.columns #Pitching
for i, value in enumerate(SuccessDataP_column_names.values):
    meansP[value] = np.mean(SuccessDataP.T.iloc[:,i])


#Create Bar Charts for Team Cluster Means
#Batting
ind = np.arange(n_components_B)  # the x locations for the groups
width = 1 # the width of the bars
fig1, ax1 = plt.subplots()
rects1 = ax1.barh(ind, meansB, width, color='#1e90ff')
ax1.set_title('Mean Number of Batters on Playoff Teams: By Type', fontsize=10)
ax1.set_yticks(ind+0.5)
ax1.set_yticklabels([])
#Pitching
indP = np.arange(n_components_P)  # the x locations for the groups
width = 1 # the width of the bars
fig2, ax2 = plt.subplots()
rects2 = ax2.barh(indP, meansP, width, color='#3cb371')
ax2.set_title('Mean Number of Pitchers on Playoff Teams: By Type', fontsize=10)
ax2.set_yticks(indP+0.5)
ax2.set_yticklabels([])


##Save Figures
#fig1.savefig('Batting_means_final.png', dpi=400, facecolor='w', edgecolor='k',
#       transparent=False, bbox_inches='tight', pad_inches=0.1)
#fig2.savefig('Pitching_means_final.png', dpi=400, facecolor='w', edgecolor='k',
#       transparent=False, bbox_inches='tight', pad_inches=0.1)
