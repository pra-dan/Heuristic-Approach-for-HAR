import numpy as np
import math
import csv
import pandas as pd

class WalkEstimation:

    #str_add = 'test1.csv'
    temp_add = 'temp_file.csv'

    def pre_STD_Calcu(str_add):
        #REMOVE ALL ZERO ENTRIES FROM ROWS
        import csv
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
        
        df = pd.read_csv(str_add)
        saved_column = df.Y #you can also use df['column_name']
        #print(saved_column)
        Y_std = np.std(saved_column, axis=0)
        #print('Y', Y_std)
        X_std = np.std(df.X)
        #print('X: ', X_std)
        return (X_std, Y_std)
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

    def WalkResult(self):
        print('Processing...reading the csv files : ')
        STD_dict = { "cocoDict_LHip.csv" : (0.0),        #PART : (STD OF X COORDINATE, STD OF Y COORDINATE)
            "cocoDict_RHip.csv" : (0,0),
            "cocoDict_LShoulder.csv" : (0,0),
            "cocoDict_LKnee.csv" : (0,0)}
        #STORING THE RESULTS IN THE ORDER MENTIONED ABOVE
        count_std = 0
        for str_add in list(STD_dict.keys()) :      #LOOPS OVER ALL THE FILENAMES STORED AS LIST OF KEYS OF THE DICTIONARY
            STD_dict[str_add] = self.STD_Calcu(str_add, self)  #USING THE "StandardDeviationTest.py"
            if((STD_dict[str_add][0])/1.75 > STD_dict[str_add][1]) :
                count_std = count_std + 1
        
        print(count_std)
        print('STD dict: ', STD_dict)
        if(count_std >= 3) :
            print('POSTURE : walking')
        print ('Next Cycle...')      
        ##END
    
    def SleepResult(SleepingOrNot, SleepingOrNot_max) :
        if(SleepingOrNot/SleepingOrNot_max > 0.75) :
            print('POSTURE : sleeping')
