import csv
import os
import pandas as pd
from pandas import DataFrame

class ClassificationHandler:

    Result_dict = { "Time" : [0],
            "Mobility" : 0,        #PART : (STD OF X COORDINATE, STD OF Y COORDINATE)
            "Posture" : 0,
            "Activity" : 0}
            

    def Write_Dict_to_CSV(self):
        with open('Classification.csv', 'a', newline='') as f_output:
            csv_output = csv.writer(f_output, delimiter=',')
            
            header = (self.Result_dict["Time"], self.Result_dict["Mobility"], self.Result_dict["Posture"], self.Result_dict["Activity"])
            print(header)
            csv_output.writerow(header)

    def Reset_Dict(self):
        #str_add in list(STD_dict.keys()) :
        print("BEFORE RESET DICT...", self.Reset_Dict) 
        for key in list(self.Result_dict.keys()):
            self.Result_dict[key] = 0
    
    #CREATING THE DICTIONARY FILE AND CREATING THE HEADER CONTAINING COLUMN NAMES
    def InitDict(self):
        #CREATING THE COLUMN HEADER OR THE FIRST ROW
        df = DataFrame(data=self.Result_dict) 
        #FIRST DELETE ANY CSV WITH THE SAME NAME
        os.remove('Classification.csv')
        #CRETING THE CSV FILE AND WRITING THE FIRST ROW TO IT
        df.to_csv(r'Classification.csv')