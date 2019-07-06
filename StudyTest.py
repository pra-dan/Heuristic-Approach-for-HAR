import math

class PopoStudyClass :
    #IMPORTANT TO DECLARE THE DICTIONARY HERE. PYTHON WILL NOT INITIALIZE IT BY DEFAULT. FOLLOW CPP HERE
    Study_Confidence_dict = {}
    def __init__(self):
        self.Study_Confidence_dict = { "NeckInclination_positive" : 0,
        "LimbInclination_positive" : 0,
        "Max_NeckInclination_LimbInclination_positives" : 0   ##DENOMINATOR FOR THE CONFIDENCE CALCULATION
        }

    #THIS WILL RESET THE DICT VALUES EACH TIME THE     
    def Reset_Study_Confidence_dict(self):
        for (key, value) in self.Study_Confidence_dict.items():
            self.Study_Confidence_dict[key] = 0  
    
    def IncrementDictValue(self,key_plus_value):
        temp = key_plus_value
        temp = temp +1
        key_plus_value = temp
        return key_plus_value

    def SlopeOf(self,p1,p2):     #THERE IS A IDENTICAL FUNCTION IN "estimator.py"
        #CALCULATING SLOPE (m) OF LINE HAVING POINTS P1 AND P2
        x1 = p1[0]
        x2 = p2[0]
        y1 = p1[1]
        y2 = p2[1]
        try:
            return (y2-y1)/(x2-x1)
        except ZeroDivisionError:
            return 0

    def StudyAnalyze(self, toPlotDict):
        #GETTING THE NECK INCLINATION NO SHOULDER DETECTED !
        neck_theta_1, neck_theta_2, neck_theta_3 = self.giveNeckInclination(toPlotDict)
        #GETTING THE LEFT AND RIGHT LIMB INCLINATION
        #POPO NOTE: By convention, this first argument is called self inside the method definition (see definition of giveArmsStudyPosture()), 
        #... but as seen in the below line, self need not be passed everytime the method/funcition is called
        #...read here: https://stackoverflow.com/questions/23944657/typeerror-method-takes-1-positional-argument-but-2-were-given
        left_limb_theta, right_limb_theta = self.giveArmsStudyPosture(toPlotDict) 
        #PRINTING THE RESULTS 
        '''
        print("NECK ANGLE 1: ", neck_theta_1)
        print("NECK ANGLE 2: ", neck_theta_2)
        print("NECK ANGLE 3: ", neck_theta_3)
        print("LEFT LIMB : ", left_limb_theta)
        print("RIGHT LIMB : ", right_limb_theta)
        print(self.Study_Confidence_dict)
        '''
        #DENOMINATOR WILL ALWAYS INCREASE WITH EVERY ITERATION. SO THAT MAX NUMBER OF POSITIVES CAN BE OBTAINED
        if neck_theta_3 != 0 and (left_limb_theta != 0 or right_limb_theta != 0) :
            self.Study_Confidence_dict["Max_NeckInclination_LimbInclination_positives"] = self.IncrementDictValue(self.Study_Confidence_dict["Max_NeckInclination_LimbInclination_positives"])
            #CHECKING FOR THE NECK INCLINCATION RESULTS TO BE POSITIVE : DENOTING STUDY POSTURE DETECTED
            if 70 <= abs(neck_theta_3) <= 90 or 25 <= abs(neck_theta_3) <= 50 :
                self.Study_Confidence_dict["NeckInclination_positive"] = self.IncrementDictValue(self.Study_Confidence_dict["NeckInclination_positive"])
            else:
                pass
            #CHECKING FOR THE LIMB INCLINCATION RESULTS TO BE POSITIVE : DENOTING STUDY POSTURE DETECTED
            #avg_limb_inclination = (left_limb_theta + right_limb_theta)/2
            #if -26 <= left_limb_theta <= -83 or 11 <= left_limb_theta <= 63  or -26 <= right_limb_theta <= -83 or 11 <= right_limb_theta <= 63 :
            if (-80 <= left_limb_theta <= -90 or 20 <= left_limb_theta <= 43 or 70 <= left_limb_theta <= 90 ) or (-80 <= right_limb_theta <= -90 or 20 <= right_limb_theta <= 43 or 70 <= right_limb_theta <= 90) :
                self.Study_Confidence_dict["LimbInclination_positive"] = self.IncrementDictValue(self.Study_Confidence_dict["LimbInclination_positive"])
            else:
                pass
          

    def StudyResult(self, res):
        print("*************************STUDY RESULTS***************************")
        print(self.Study_Confidence_dict)
        neck = self.Study_Confidence_dict["NeckInclination_positive"]
        limb = self.Study_Confidence_dict["LimbInclination_positive"]
        maX = self.Study_Confidence_dict["Max_NeckInclination_LimbInclination_positives"]
        try:
            if(neck/maX > 0.25 and limb/maX > 0.40) :   #CONSIDERING 25% NECK CONFIDENCE AND 40% LIMB CONFIENCE TO BE SUFFICIENT TO CLASSIFY THIS POSTURE
                print('Activity : Studying')

            elif(neck/maX <= 0.25 and limb/maX > 0.40):  #THAT IS, ONLY LIMB ESIMATES "Studying" WHEREAS NECK DOES NOT   
                #GET AUDIO CONFIRMATION
                from speechToText_test import SpeechText 
                answer = SpeechText.AskAndListen(SpeechText, 'Are You Studying right now?')
                if(answer == 'yes') :
                    print('Activity : Studying')        
                    res.Result_dict["Activity"] = "Studying"             
            else:
                pass
        except :
            pass
            

              

    def giveNeckInclination(self,toPlotDict):
        #flag_temp = 1
        #FIND SLOPE OF LINES JOINING ANY AVAILABLE PART ON THE HEAD
        ##TRYING WITH THE LEFT PARTS/KEYPOINTS
        try:
            if toPlotDict["Nose"] != (0,0) and toPlotDict["LShoulder"] != (0,0) and toPlotDict["LHip"] != (0,0):
                m1 = self.SlopeOf(toPlotDict["Nose"], toPlotDict["LShoulder"])
                m2 = self.SlopeOf(toPlotDict["LHip"], toPlotDict["LShoulder"])
                neck_theta_1 = math.degrees(math.atan((m2-m1)/(1+m2*m1)))
            else:
                neck_theta_1 = 0
                #pass
        except: #IN CASE OF ANY ERROR
            #DO NOTHING AND TRY AGAIN IN THE NEXT ITERATION
            neck_theta_1 = 0
        ##TRYING WITH THE RIGHT PARTS/KEYPOINTS  
        # POPO NOTE : DON'T PASS self AS THE ARGUMENT WHILE CALLING METHODS FROM WITHIN METHODS. 
        # ...SIMPLY, DECLARE self AS THE FIRST ARGUMENT WHILE DEFINING THE METHODS  
        try:
            if toPlotDict["Nose"] != (0,0) and toPlotDict["RShoulder"] != (0,0) and toPlotDict["RHip"] != (0,0):
                m1 = self.SlopeOf(toPlotDict["Nose"], toPlotDict["RShoulder"])
                m2 = self.SlopeOf(toPlotDict["RHip"], toPlotDict["RShoulder"])
                neck_theta_2 = math.degrees(math.atan((m2-m1)/(1+m2*m1)))
            else:
                neck_theta_2 = 0
                #pass
        except:
            #DO NOTHING AND TRY AGAIN IN THE NEXT ITERATION
            neck_theta_2 = 0

        try:
            if toPlotDict["LShoulder"] != (0,0):
                avl_shoulder = toPlotDict["LShoulder"]
            elif toPlotDict["RShoulder"] != (0,0):
                avl_shoulder = toPlotDict["RShoulder"]
            else:
                pass

            if toPlotDict["LHip"] != (0,0):
                avl_Hip = toPlotDict["LHip"]
            elif toPlotDict["RHip"] != (0,0):
                avl_Hip = toPlotDict["RHip"]
            else:
                pass
            
            m1 = self.SlopeOf(toPlotDict["Nose"], avl_shoulder)
            m2 = self.SlopeOf(avl_Hip, avl_shoulder)
            neck_theta_3 = math.degrees(math.atan((m2-m1)/(1+m2*m1)))
        except:
            neck_theta_3 = 0
        
        return neck_theta_1, neck_theta_2, neck_theta_3


    def giveArmsStudyPosture(self, toPlotDict):
        #AIM IS TO FIND THE ANGLE OF THE ARM AND FOREARM (XSHOULDER - XELBOW - XWRIST)
        #arm_flag = 1 #ALL FINE
        try:
            if toPlotDict["LShoulder"] != (0,0) and toPlotDict["LElbow"] != (0,0) and toPlotDict["LWrist"] != (0,0): 
               #THAT IS , ALL PARTS ARE AVAILABLE
               m1 = self.SlopeOf(toPlotDict["LElbow"], toPlotDict["LShoulder"]) 
               m2 = self.SlopeOf(toPlotDict["LElbow"], toPlotDict["LWrist"])
               left_limb_theta = math.degrees(math.atan((m2-m1)/(1+m2*m1)))
            
            else:
                #DO NOTHING AND TRY AGAIN IN THE NEXT ITERATION
                left_limb_theta = 0
        except :
            left_limb_theta = 0
        #DOING THE SAME FOR THE OTHER LIMB    
        try:
            if toPlotDict["RShoulder"] != (0,0) and toPlotDict["RElbow"] != (0,0) and toPlotDict["RWrist"] != (0,0): 
               #THAT IS , ALL PARTS ARE AVAILABLE
               m1 = self.SlopeOf(toPlotDict["RElbow"], toPlotDict["RShoulder"]) 
               m2 = self.SlopeOf(toPlotDict["RElbow"], toPlotDict["RWrist"])
               right_limb_theta = math.degrees(math.atan((m2-m1)/(1+m2*m1)))
            
            else:
                #DO NOTHING AND TRY AGAIN IN THE NEXT ITERATION
                right_limb_theta = 0
        except :
            right_limb_theta = 0
        
        return left_limb_theta, right_limb_theta
