#from tf_pose.estimator import TfPoseEstimator
import time
#JUST FOR PLOTTING PURPOSE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

#dist_dict has all the body parts detected. IMPORTING IT.
dist_dict = __import__('tf_pose.estimator').TfPoseEstimator.dist_dict 
#THE COORDINATES OF EACH PART CAN BE ACCESSED BY REFERRING TO THIS CLASS: CocoPart
'''
This should help to relate the part number with the part name
        class CocoPart(Enum):
            Nose = 0
            Neck = 1
            RShoulder = 2
            RElbow = 3
            RWrist = 4
            LShoulder = 5
            LElbow = 6
            LWrist = 7
            RHip = 8
            RKnee = 9
            RAnkle = 10
            LHip = 11
            LKnee = 12
            LAnkle = 13
            REye = 14
            LEye = 15
            REar = 16
            LEar = 17
            Background = 18

            ...The syntax is: cocoDict[CocoPart.LHip]
'''
def isWalking(cocoDict):
    torsoDict = {
        "LShoulder" : (0,0),
        "LHip" : (0,0),
        "RHip" : (0,0),
        "LKnee" : (0,0)
    }
    #TRYING TO GET THE COORDINATES OF THE TORSO
    print('ENTERED WALKeSTIMATION FN...')
    
    for i in range(4) :
        try:
            torsoDict["LShoulder"] = cocoDict[CocoPart.LShoulder]
        except KeyError:
            pass
        ##print('Not There')

        try:
            torsoDict["LHip"] = cocoDict[CocoPart.LHip]
        except KeyError:
            pass

        style.use('fivethirtyeight')
        fig = plt.figure()

        ax1 = fig.add_subplot(1,1,1)
        ax1.clear()
        ax1.plot(torsoDict["LShoulder"][1], torsoDict["LShoulder"][2])
        sleep(0.5)

    
    