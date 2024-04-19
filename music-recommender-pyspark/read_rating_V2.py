#!/usr/bin/env python

import numpy

dataDir='/YOUR_DIRECTORY/Project/data_in_matrixForm/'
file_name_test=dataDir + 'testTrack_hierarchy.txt'
file_name_train=dataDir + 'trainIdx2_matrix.txt'
output_file= dataDir + 'output1.txt'

fTest= open(file_name_test, 'r')
fTrain=open(file_name_train, 'r')
Trainline= fTrain.readline()
fOut = open(output_file, 'w')

trackID_vec=[0]*6
albumID_vec=[0]*6
artistID_vec=[0]*6
lastUserID=-1

user_rating_inTrain=numpy.zeros(shape=(6,3))

for line in fTest:
	arr_test=line.strip().split('|')
	userID= arr_test[0]
	trackID= arr_test[1]
	albumID= arr_test[2]
	artistID=arr_test[3]

	if userID!= lastUserID:
		ii=0
		user_rating_inTrain=numpy.zeros(shape=(6,3))

	trackID_vec[ii]=trackID
	albumID_vec[ii]=albumID
	artistID_vec[ii]=artistID
	ii=ii+1
	lastUserID=userID

	if ii==6 : 
		while (Trainline):
		# for Trainline in fTrain:
			arr_train = Trainline.strip().split('|')
			trainUserID=arr_train[0]
			trainItemID=arr_train[1]
			trainRating=arr_train[2]
			Trainline=fTrain.readline()		

			if trainUserID< userID:
				continue
			if trainUserID== userID:				
				for nn in range(0, 6):
					if trainItemID==albumID_vec[nn]:
						user_rating_inTrain[nn, 0]=trainRating
					if trainItemID==artistID_vec[nn]:
						user_rating_inTrain[nn, 1]=trainRating
			if trainUserID> userID:
				for nn in range(0, 6):
					outStr=str(userID) + '|' + str(trackID_vec[nn])+ '|' + str(user_rating_inTrain[nn,0]) + '|' + str(user_rating_inTrain[nn, 1])
					fOut.write(outStr + '\n')
				break



fTest.close()
fTrain.close()



