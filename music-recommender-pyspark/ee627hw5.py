# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:39:20 2022

@author: camer
"""

import pandas as pd
import numpy as np



def userData(ftest, ftrain):
    
    # headers for training columns
    training = ['UserId', 'TrackId', 'Score']
    # read train file and cast all data as ints
    train = pd.read_csv(ftrain, header=None, delimiter="|", names = training,
                        dtype={'UserId':int,'TrackId':int,'Score':int})
    # make test dataframe and convert filled columns to ints
    testing = ['UserId','TrackId','AlbumId','ArtistId']
    test = pd.read_csv(ftest, header=None, delimiter="|", names = testing,
            dtype={'UserId':int,'TrackId':int})
    # replace missing artist and album ids with 0 and convert to ints
    test['AlbumId']=pd.to_numeric(test['AlbumId'],errors='coerce').fillna(value=0).astype(int)
    test['ArtistId']=pd.to_numeric(test['ArtistId'],errors='coerce').fillna(value=0).astype(int)
    # read test file with the context manager to check for genre data
    with open(ftest, 'r') as f:
        genres = []
        for l in f.readlines():
            # if genre data exists, format list as ints and add to genres list
            if len(l.split("|")) > len(testing):
                g = set(map(lambda x: int(x.strip()), l.split('|')[4:]))
                genres.append(g)
            # else set genre as 'None'
            else:
                genres.append(set())
    # add genre data to test dataframe
    test['GenreId'] = genres
    
    hierarchy = pd.DataFrame({'TrackId': test.TrackId, 
            'Hierarchy': tuple(zip(test.AlbumId,test.ArtistId,test.GenreId))})
    
    hierarchy.drop_duplicates(subset="TrackId",keep='first', inplace=True)
    
    tracks = set(hierarchy.TrackId.unique())
    
    return tracks, hierarchy, test.groupby('UserId'), train.groupby('UserId')

def submission(users):

    # filename = 'hw5submission.csv'
    
    ids = test['UserId'].astype(str)+'-'+test['TrackId'].astype(str)
    
    pred = []
    
    df = pd.DataFrame({'TrackID': ids, 'Prediction':pred})
    
    return df#df.to_csv(filename, index=False)

def similarity(tuple1, tuple2):
    tuple2 = tuple([tuple2[0]])+tuple2[1]
    # check if album or artist is the same; 0 or 100% similar
    albumSim = len({tuple1[1]}.intersection({tuple2[1]})) if tuple1[1] != 0 else 0
    artistSim = len({tuple1[2]}.intersection({tuple2[2]})) if tuple1[2] != 0 else 0
    # check % similarity of genre sets
    if len(tuple1[-1]) >0 and len(tuple2[-1]) >0:
        genreSim = len(tuple1[-1].intersection(tuple2[-1]))/len(tuple1[-1].union(tuple2[-1]))
    else:
        genreSim = 0
    
    union = ((albumSim + artistSim*0.8 + genreSim) / 3) *100
    
    return union

def getSimRatings(user_tracks, user_rated, tracks):
    r = []    
    for t in user_tracks:
        rated = {}
        for u in user_rated:
            if u in tracks:
                # similarity between a test track and user rated track that is also in the test set
                sim = similarity(t, hierarchy.loc[hierarchy['TrackId']==u].head(1).to_numpy().flatten())
                if sim > 0:
                    rated[u] = sim
        r.append(rated)
    
    return r

def getClassifications(r,user_rated):
    
    s = [0,0,0,0,0,0]
    
    for i in range(6):
        if len(r[i]) > 0:
            mx = max(r[i].values())
            s[i] = np.mean([user_rated[k] for k,v in r[i].items() if v == mx])
    # threshold for rating
    t = 60
    p = (np.array(s) > t).astype(int)
        
    return p

if __name__ == '__main__':
    
    filepath = "../ee627/YahooMusic_V2_1/"
    
    ftrain = filepath + 'trainIdx2_matrix.txt'    
    ftest = filepath + 'testTrack_hierarchy.txt'
    
    tracks, hierarchy, test, train = userData(ftest, ftrain)
    # choose number of users to calc recommendations for
    n = 5
    userList = tuple(test.groups.keys())
    users =	userList[0:2]
    # make predictions    
    predictions = []
    
    for u in users:
        x,y = train.get_group(u).drop('UserId',axis=1), test.get_group(u).drop('UserId',axis=1)
        
        user_rated = {a:b for a,b in zip(x.TrackId,x.Score)}
        user_tracks = tuple(zip(y.TrackId,y.AlbumId,y.ArtistId,y.GenreId))
        
        r = getSimRatings(user_tracks, user_rated, tracks)
        
        predictions.extend(getClassifications(r,user_rated))
    
    # for i in range(n):
    #     ratio = sum(predictions[0+i*6:5+i*6])
    #     if ratio != 3:
    #         print(userList[i],ratio)
            
    # *** to sum in 1 min, need 2k predictions per second ***
    
    