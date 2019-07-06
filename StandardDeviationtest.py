import numpy as np
import math
import csv
import pandas as pd
import os

class PositionEstimation:

    #str_add = 'test1.csv'
    temp_add = 'temp_file.csv'

    def pre_STD_Calcu(str_add):
        #REMOVE ALL ZERO ENTRIES FROM ROWS
        #import csv         ##IT PROBABLY CAUSED "ImportError" !!
        list_new = []
        with open(str_add, 'r') as f:
            lines = f.readlines()
        with open(str_add, 'w') as f:
            for line in lines:
                if line.strip("\n") != '0,0':
                    f.write(line)
        
        #REMOVES ALL EMPTY ENTRIES FROM ROWS BUT INTO A TEMPORARY FILE 
        #AS PER THIS LINK: "https://stackoverflow.com/questions/32161216/python-csv-delete-empty-rows?lq=1"
        with open(str_add) as input, open('temp_file.csv', 'w') as output:
            non_blank = (line for line in input if line.strip())
            output.writelines(non_blank)

        #NOW LETS COPY THE CONTENTS BACK TO THE ORIGINAL FILE
        from shutil import copyfile
        copyfile('temp_file.csv',str_add )
    

    #CALCULATE STD AND VARIANCE 
    def STD_Calcu(str_add, self) :
        self.pre_STD_Calcu(str_add)
        #NOW WE CAN CALCULATE THE STD 
        #read csv file ,SAVE THE Y-AXIS COLUMN IN an array
        
        try:
            df = pd.read_csv(str_add)
            saved_column = df['Y'] #you can also use df['column_name']
            #print("saved_column: ",saved_column)
            Y_std = np.std(saved_column, axis=0)
            #print('Y', Y_std)
            X_std = np.std(df.X)
            #print('X: ', X_std)
            return (X_std, Y_std)
        except :
            #print("FILE IS EMPTY")    #IN ORDER TO AVOID ERRORS LIKE "pandas.errors.EmptyDataError: No columns to parse from file"
            return (0,0)
        
    #Uncomment this to run this file manually on a local csv file
    #STD_Calcu(str_add)

    def CSV_for_Walking_Init():
        myFile1 = open('cocoDict_LHip.csv', 'w')
        myFile2 = open('cocoDict_RHip.csv', 'w')
        myFile3 = open('cocoDict_LKnee.csv', 'w')
        myFile4 = open('cocoDict_LShoulder.csv', 'w')
        with myFile1:
            writer1 = csv.writer(myFile1)
            writer1.writerow('XY')

        with myFile2:
            writer2 = csv.writer(myFile2)
            writer2.writerow('XY')
            
        with myFile3:
            writer3 = csv.writer(myFile3)
            writer3.writerow('XY')

        with myFile4:
            writer4 = csv.writer(myFile4)
            writer4.writerow('XY') 

    def CSV_for_Walking_Def(toPlotDict):
        myFile1 = open('cocoDict_LHip.csv', 'a')
        myFile2 = open('cocoDict_RHip.csv', 'a')
        myFile3 = open('cocoDict_LKnee.csv', 'a')
        myFile4 = open('cocoDict_LShoulder.csv', 'a')
        #BASED ON THE SUGGESTION HERE: "https://stackoverflow.com/questions/2363731/append-new-row-to-old-csv-file-python"
                
        with myFile1:
            writer1 = csv.writer(myFile1)
            writer1.writerow(toPlotDict["LHip"])
                
        with myFile2:
            writer2 = csv.writer(myFile2)
            writer2.writerow(toPlotDict["RHip"])
            
        with myFile3:
            writer3 = csv.writer(myFile3)
            writer3.writerow(toPlotDict["LKnee"])

        with myFile4:
            writer4 = csv.writer(myFile4)
            writer4.writerow(toPlotDict["LShoulder"])
        
        

    def WalkResult(self, res):
        #print('Processing...reading the csv files : ')
        STD_dict = { "cocoDict_LHip.csv" : (0.0),        #PART : (STD OF X COORDINATE, STD OF Y COORDINATE)
            "cocoDict_RHip.csv" : (0,0),
            "cocoDict_LShoulder.csv" : (0,0),
            "cocoDict_LKnee.csv" : (0,0)}
        #STORING THE RESULTS IN THE ORDER MENTIONED ABOVE
        count_std = 0
        
        for str_add in list(STD_dict.keys()) :      #LOOPS OVER ALL THE FILENAMES STORED AS LIST OF KEYS OF THE DICTIONARY
            STD_dict[str_add] = self.STD_Calcu(str_add, self)  #USING THE "StandardDeviationTest.py"
            #GOT THE STANDARD DEVIATION FOR THE ENTRIES IN THE CSV FILE BEING ITERATED OVER
            #print("STD = ",STD_dict[str_add])
            if((STD_dict[str_add][0])/1.75 > STD_dict[str_add][1]) :
                count_std = count_std + 1
            
        #print("count_std : ",count_std)
        #print('STD dict: ', STD_dict)
        if(count_std > 3) :    #THAT IS CONSIDERING 75% CONFIDENCE (3/4)
            print('Mobility : Walking')
            res.Result_dict["Mobility"] = "Walking"
        elif(count_std>0):
            print('Mobility : Still')
            res.Result_dict["Mobility"] = "Still"
        else:
            res.Result_dict["Mobility"] = "0"
        #print ('Next Cycle...') 
            
        ##END try
    
    def RemoveFiles(self):
        STD_dict = { "cocoDict_LHip.csv" : (0.0),        #PART : (STD OF X COORDINATE, STD OF Y COORDINATE)
            "cocoDict_RHip.csv" : (0,0),
            "cocoDict_LShoulder.csv" : (0,0),
            "cocoDict_LKnee.csv" : (0,0)}
        for str_add in list(STD_dict.keys()) :
            os.remove(str_add)

    def SleepResult(self, Posture_Confidence_dict, res) :
        try:
            SleepingOrNot = Posture_Confidence_dict["SleepingOrNot"]
            SittingOrNot = Posture_Confidence_dict["SittingOrNot"]
            StandingOrNot = Posture_Confidence_dict["StandingOrNot"]
            
            SleepingOrNot_max = Posture_Confidence_dict["Max_Stand_Sleep_Sitting"]
            if(SleepingOrNot/SleepingOrNot_max > 0.75) :
                print('POSTURE : Lying')
                res.Result_dict["Posture"] = "Lying"
            
            elif(SittingOrNot/SleepingOrNot_max > 0.75) :
                print('POSTURE : Sitting')
                res.Result_dict["Posture"] = "Sitting"
            
            elif(StandingOrNot/SleepingOrNot_max > 0.75) :
                print('POSTURE : Standing')
                res.Result_dict["Posture"] = "Standing"
            else:
                pass

        except :
            pass
