#!/usr/bin/env python

import numpy as np
import os
from collections import defaultdict

SCRIPT = os.path.dirname(__file__)
file_name_test = os.path.join(SCRIPT, "data", "testTrack_hierarchy.txt")
file_name_train = os.path.join(SCRIPT, "data", "trainIdx2_matrix.txt")
output_file = os.path.join(SCRIPT, "data", "out2.txt")

# Load train data into a dictionary for quick lookup
def load_train_data(file_name_train):
    train_data = defaultdict(lambda: {'albums': defaultdict(float), 'artists': defaultdict(float), 'genres': defaultdict(float)})
    with open(file_name_train, 'r') as fTrain:
        for line in fTrain:
            arr_train = line.strip().split('|')
            trainUserID = arr_train[0]
            trainItemID = arr_train[1]
            trainRating = float(arr_train[2])
            train_data[trainUserID][trainItemID] = trainRating
    return train_data

train_data = load_train_data(file_name_train)

# Process the test data
with open(file_name_test, 'r') as fTest, open(output_file, 'w') as fOut:
    lastUserID = None
    trackID_vec = [0] * 6
    albumID_vec = [0] * 6
    artistID_vec = [0] * 6
    genreID_vec = [[] for _ in range(6)]
    user_rating_inTrain = np.zeros((6, 3))  # Adjust array for album, artist, and genre ratings
    i = 0  # Initialize index variable

    for line in fTest:
        arr_test = line.strip().split('|')
        userID = arr_test[0]
        trackID = arr_test[1]
        albumID = arr_test[2] if arr_test[2] != 'None' else None
        artistID = arr_test[3] if arr_test[3] != 'None' else None
        genreIDs = arr_test[4:] if len(arr_test) > 4 else []

        if userID != lastUserID:
            if lastUserID is not None:
                # Output the ratings for the previous user
                for nn in range(6):
                    genre_ratings = '|'.join(map(str, user_rating_inTrain[nn, 2:]))
                    outStr = f"{lastUserID}|{trackID_vec[nn]}|{user_rating_inTrain[nn, 0]}|{user_rating_inTrain[nn, 1]}|{genre_ratings}"
                    fOut.write(outStr + '\n')
            # Reset for the new user
            i = 0
            user_rating_inTrain = np.zeros((6, 2 + len(genreIDs)))  # Reset the array and adjust for genres

        trackID_vec[i] = trackID
        albumID_vec[i] = albumID
        artistID_vec[i] = artistID
        genreID_vec[i] = genreIDs
        i += 1
        lastUserID = userID

        if i == 6:
            user_ratings = train_data.get(userID, {})
            for nn in range(6):
                if albumID_vec[nn] is not None:
                    user_rating_inTrain[nn, 0] = user_ratings.get(albumID_vec[nn], 0)
                if artistID_vec[nn] is not None:
                    user_rating_inTrain[nn, 1] = user_ratings.get(artistID_vec[nn], 0)
                for idx, genreID in enumerate(genreID_vec[nn]):
                    user_rating_inTrain[nn, 2 + idx] = user_ratings.get(genreID, 0)
            # Output the ratings for the current user
            for nn in range(6):
                genre_ratings = '|'.join(map(str, user_rating_inTrain[nn, 2:]))
                outStr = f"{userID}|{trackID_vec[nn]}|{user_rating_inTrain[nn, 0]}|{user_rating_inTrain[nn, 1]}|{genre_ratings}"
                fOut.write(outStr + '\n')
            i = 0  # Reset index after processing

    # Output any remaining ratings for the last user
    if lastUserID is not None and i > 0:
        for nn in range(i):
            genre_ratings = '|'.join(map(str, user_rating_inTrain[nn, 2:]))
            outStr = f"{lastUserID}|{trackID_vec[nn]}|{user_rating_inTrain[nn, 0]}|{user_rating_inTrain[nn, 1]}|{genre_ratings}"
            fOut.write(outStr + '\n')

fTest.close()
fOut.close()
