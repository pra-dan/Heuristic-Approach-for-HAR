#python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0
#JUST FOR RUNNING PLOTS FOR WALK ESTIMATION
#-> CAPABLE OF PLOTTING ONE OR MORE COORDINATES OF SELECTIVE BODY PART KEYPOINTS RECORDED WHILE MOVEMENT
#-> CAPABLE OF CALCULATING THE STD AFTER PRE-PROCSESSING THE CSV FILES
import argparse
import logging
import time
#POPO EDIT
import matplotlib
import csv
import datetime
import time

from tf_pose.estimator  import POPOclass
from StandardDeviationtest import PositionEstimation

#from collections import OrderedDict
#csv.register_dialect('myDialect', delimiter=')', quoting=csv.QUOTE_NONE)
#INITIALIZING THE CSV FILES AND THEIR HANDLERS
PositionEstimation.CSV_for_Walking_Init()
##END

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
##POPO EDIT

##END
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    ##logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    #POPO EDIT
    LHip_list = []
    RHip_list = []
    LShoulder_list = []
    LKnee_list = []
    ret_val, image = cam.read()
    #count = 0

    
    ##END
    while True: 
        #POPO EDIT
        timer = 15 #CHANGE IN SECONDS
        #AS PER THIS LINK : "https://stackoverflow.com/questions/24374620/python-loop-to-run-for-certain-amount-of-seconds"
        t_end = time.time() + timer
        print('Capturing data for : ', timer)
        ##END
        while time.time() < t_end:
            #POPO EDIT
            #DISPLAY DATE AND TIME
            datet = str(datetime.datetime.now())
            ret_val, image = cam.read()
            print('TIME : ', datet)
                       
            ##END
            ##logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

            ##logger.debug('postprocess+')\

            #POPO EDIT
            #image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            SleepingOrNot_max = 0
            image, SleepingOrNot, cocoDict_clone  = TfPoseEstimator.draw_humans(image, humans, imgcopy=False, SleepingOrNot_max)
            #FETCHING THE REQUIRED VARs FROM 'TfPoseEstimator.draw_humans' SO AND PROVIDING THEM TO THE FUNCTION 'POPOclass.isWalking', INORDER
            #TO AVOID CIRCULAR DEPENDENCIES
            toPlotDict = POPOclass.isWalking(cocoDict_clone)
            
            #EDITING CSV FILES AND STORING RECURDED DATA
            PositionEstimation.CSV_for_Walking_Def(toPlotDict)
            
            ##logger.debug('show+')
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            key = cv2.waitKey(1) & 0xFF
            
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        #POPO EDIT
        #DATA ACQUIRED. DISPLAYING RESULTS FOR WALKING
        PositionEstimation.WalkResult(PositionEstimation)
        #DISPLAYING RESULTS FOR SLEEPING/LYING
        PositionEstimation.SleepResult(SleepingOrNot, SleepingOrNot_max)


        

    cv2.destroyAllWindows()
