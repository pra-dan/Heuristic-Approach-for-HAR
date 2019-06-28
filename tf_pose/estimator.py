import logging
import math

#POPO EDIT
import matplotlib
'''
from matplotlib import pyplot
#JUST FOR PLOTTING PURPOSE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
'''
###END

import slidingwindow as sw

import cv2
import numpy as np
import tensorflow as tf
import time

from tf_pose import common
from tf_pose.common import CocoPart
from tf_pose.tensblur.smoother import Smoother

try:
    from tf_pose.pafprocess import pafprocess
except ModuleNotFoundError as e:
    print(e)
    print('you need to build c++ library for pafprocess. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess')
    exit(-1)

logger = logging.getLogger('TfPoseEstimator')
logger.handlers.clear()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

#POPO EDITpyplot
def distance(p1,p2):
    # Calculating distance 
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    return math.sqrt(math.pow(x2 - x1, 2) +  math.pow(y2 - y1, 2) * 1.0)

def slopeOf(p1,p2):
    #CALCULATING SLOPE (m) OF LINE HAVING POINTS P1 AND P2
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    try:
        return (y2-y1)/(x2-x1)
    except ZeroDivisionError:
        return 0

class POPOclass :
    def isWalking(cocoDict):
        #USING THESE 4 KEYPOINTS TO ESTIMATE THE WALK
        torsoDict = {
            "LShoulder" : (0,0),
            "LHip" : (0,0),
            "RHip" : (0,0),
            "LKnee" : (0,0)
        }
        #TRYING TO GET THE COORDINATES OF THE TORSO
        
        try:
            torsoDict["LShoulder"] = cocoDict[CocoPart.LShoulder]

        except KeyError:
            torsoDict["LShoulder"] = (0,0)
            pass
            ##print('Not There')

        try:
            torsoDict["LHip"] = cocoDict[CocoPart.LHip]
        except KeyError:
            torsoDict["LHip"] = (0,0)
            pass
        
        try:
            torsoDict["RHip"] = cocoDict[CocoPart.RHip]
        except KeyError:
            torsoDict["RHip"] = (0,0)
            pass
        
        try:
            torsoDict["LKnee"] = cocoDict[CocoPart.LKnee]
        except KeyError:
            torsoDict["LKnee"] = (0,0)
            pass


        return torsoDict

    
    

###END

def _round(v):
    return int(round(v))


def _include_part(part_list, part_idx):
    for part in part_list:
        if part_idx == part.part_idx:
            return True, part
    return False, None


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def get_face_box(self, img_w, img_h, mode=0):
        """
        Get Face box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :param mode:
        :return:
        """
        # SEE : https://github.com/ildoonet/tf-pose-estimation/blob/master/tf_pose/common.py#L13
        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _REye = CocoPart.REye.value
        _LEye = CocoPart.LEye.value
        _REar = CocoPart.REar.value
        _LEar = CocoPart.LEar.value

        _THRESHOLD_PART_CONFIDENCE = 0.2
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]

        is_nose, part_nose = _include_part(parts, _NOSE)
        if not is_nose:
            return None

        size = 0
        is_neck, part_neck = _include_part(parts, _NECK)
        if is_neck:
            size = max(size, img_h * (part_neck.y - part_nose.y) * 0.8)

        is_reye, part_reye = _include_part(parts, _REye)
        is_leye, part_leye = _include_part(parts, _LEye)
        if is_reye and is_leye:
            size = max(size, img_w * (part_reye.x - part_leye.x) * 2.0)
            size = max(size,
                       img_w * math.sqrt((part_reye.x - part_leye.x) ** 2 + (part_reye.y - part_leye.y) ** 2) * 2.0)

        if mode == 1:
            if not is_reye and not is_leye:
                return None

        is_rear, part_rear = _include_part(parts, _REar)
        is_lear, part_lear = _include_part(parts, _LEar)
        if is_rear and is_lear:
            size = max(size, img_w * (part_rear.x - part_lear.x) * 1.6)

        if size <= 0:
            return None

        if not is_reye and is_leye:
            x = part_nose.x * img_w - (size // 3 * 2)
        elif is_reye and not is_leye:
            x = part_nose.x * img_w - (size // 3)
        else:  # is_reye and is_leye:
            x = part_nose.x * img_w - size // 2

        x2 = x + size
        if mode == 0:
            y = part_nose.y * img_h - size // 3
        else:
            y = part_nose.y * img_h - _round(size / 2 * 1.2)
        y2 = y + size

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
            return None
        if mode == 0:
            return {"x": _round((x + x2) / 2),
                    "y": _round((y + y2) / 2),
                    "w": _round(x2 - x),
                    "h": _round(y2 - y)}
        else:
            return {"x": _round(x),
                    "y": _round(y),
                    "w": _round(x2 - x),
                    "h": _round(y2 - y)}

    def get_upper_body_box(self, img_w, img_h):
        """
        Get Upper body box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :return:
        """

        if not (img_w > 0 and img_h > 0):
            raise Exception("img size should be positive")

        _NOSE = CocoPart.Nose.value

        _NECK = CocoPart.Neck.value
        _RSHOULDER = CocoPart.RShoulder.value
        _LSHOULDER = CocoPart.LShoulder.value
        _THRESHOLD_PART_CONFIDENCE = 0.3
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]
        part_coords = [(img_w * part.x, img_h * part.y) for part in parts if
                       part.part_idx in [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]]

        if len(part_coords) < 5:
            return None

        # Initial Bounding Box
        x = min([part[0] for part in part_coords])
        y = min([part[1] for part in part_coords])
        x2 = max([part[0] for part in part_coords])
        y2 = max([part[1] for part in part_coords])

        # # ------ Adjust heuristically +
        # if face points are detcted, adjust y value

        is_nose, part_nose = _include_part(parts, _NOSE)
        is_neck, part_neck = _include_part(parts, _NECK)
        torso_height = 0
        if is_nose and is_neck:
            y -= (part_neck.y * img_h - y) * 0.8
            
            torso_height = max(0, (part_neck.y - part_nose.y) * img_h * 2.5)
        #
        # # by using shoulder position, adjust width
        is_rshoulder, part_rshoulder = _include_part(parts, _RSHOULDER)
        is_lshoulder, part_lshoulder = _include_part(parts, _LSHOULDER)
        if is_rshoulder and is_lshoulder:
            half_w = x2 - x
            dx = half_w * 0.15
            x -= dx
            x2 += dx
        elif is_neck:
            if is_lshoulder and not is_rshoulder:
                half_w = abs(part_lshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)
            elif not is_lshoulder and is_rshoulder:
                half_w = abs(part_rshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)

        # ------ Adjust heuristically -

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
            return None
        return {"x": _round((x + x2) / 2),
                "y": _round((y + y2) / 2),
                "w": _round(x2 - x),
                "h": _round(y2 - y)}

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()


class PoseEstimator:
    def __init__(self):
        pass

    @staticmethod
    def estimate_paf(peaks, heat_mat, paf_mat):
        pafprocess.process_paf(peaks, heat_mat, paf_mat)

        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False

            for part_idx in range(18):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
                    float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0],
                    pafprocess.get_part_score(c_idx)
                )

            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

        return humans


class TfPoseEstimator:
    # TODO : multi-scale

    def __init__(self, graph_path, target_size=(320, 240), tf_config=None):
        self.target_size = target_size

        # load graph
        logger.info('loading graph from %s(default size=%dx%d)' % (graph_path, target_size[0], target_size[1]))
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='TfPoseEstimator')
        self.persistent_sess = tf.Session(graph=self.graph, config=tf_config)

        # for op in self.graph.get_operations():
        #     print(op.name)
        # for ts in [n.name for n in tf.get_default_graph().as_graph_def().node]:
        #     print(ts)

        self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/image:0')
        self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')
        self.tensor_heatMat = self.tensor_output[:, :, :, :19]
        self.tensor_pafMat = self.tensor_output[:, :, :, 19:]
        self.upsample_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='upsample_size')
        self.tensor_heatMat_up = tf.image.resize_area(self.tensor_output[:, :, :, :19], self.upsample_size,
                                                      align_corners=False, name='upsample_heatmat')
        self.tensor_pafMat_up = tf.image.resize_area(self.tensor_output[:, :, :, 19:], self.upsample_size,
                                                     align_corners=False, name='upsample_pafmat')
        smoother = Smoother({'data': self.tensor_heatMat_up}, 25, 3.0)
        gaussian_heatMat = smoother.get_output()

        max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
        self.tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                                     tf.zeros_like(gaussian_heatMat))

        self.heatMat = self.pafMat = None

        # warm-up
        self.persistent_sess.run(tf.variables_initializer(
            [v for v in tf.global_variables() if
             v.name.split(':')[0] in [x.decode('utf-8') for x in
                                      self.persistent_sess.run(tf.report_uninitialized_variables())]
             ])
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1], target_size[0]]
            }
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1] // 2, target_size[0] // 2]
            }
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1] // 4, target_size[0] // 4]
            }
        )

        # logs
        if self.tensor_image.dtype == tf.quint8:
            logger.info('quantization mode enabled.')

    def __del__(self):
        # self.persistent_sess.close()
        pass

    def get_flops(self):
        flops = tf.profiler.profile(self.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        return flops.total_float_ops

    @staticmethod
    def _quantize_img(npimg):
        npimg_q = npimg + 1.0
        npimg_q /= (2.0 / 2 ** 8)
        # npimg_q += 0.5
        npimg_q = npimg_q.astype(np.uint8)
        return npimg_q

    @staticmethod
    def draw_humans(npimg, humans, imgcopy=False, SleepingOrNot_max):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}
################################################################
        #POPO EDIT
        #print("ENTERED ESTIMATOR.PY ..... DOING THE MAGIC")
        #matplotlib.use('TKAgg',warn=False, force=True)
        #import matplotlib.pyplot as plt1
        #fig1 = plt1.figure()
        xlist = []
        ylist = []
        #IT IS VERY IMPORTANT TO DEFINE THEM AS GLOBAL OUTSIDE THE LOOP (SCOPE) WHERE THEY ARE BEING MANIPULATED.
        #THIS ENABLES US TO RETURN THEM ONCE THEY HAVE BEEN MODIFIED/ USED
        cocoBind = []
        #CREATING A DICTIONARY DIST OF EACH PART FROM NOSE : NAME OF THE PART       
        #DECLARED IT GLOBAL DUE TO ISSUE DISCUSSED HERE (https://stackoverflow.com/questions/42861989/python-how-do-you-initialize-a-global-variable-only-when-its-not-defined)
           
        global dist_dict 
        dist_dict = {}

        global cocoDict
        cocoDict = {}
        ###END

        for human in humans:
            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
 ###################################################################################################               
                #POPO EDIT
                '''This should help to relate the part number with the part name
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
                '''
 
                xlist.append(center[0])
                ylist.append(center[1])
                #ANNOTATE PLACED HERE BECAUSE IT IS AN ITERATIVE PROCESS AND CANNOT BE DONE AFTER THE COMPLETE... 
                #...LIST HAS ACCUMULATED
                ##plt1.annotate(s=(center[0],center[1],CocoPart(i)),xy=(center[0],center[1]))
                
                cocoBind.append(CocoPart(i))
                #print(CocoPart[1])
                ###END

                centers[i] = center
                cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
#######################################################################################################################
            #POPO EDIT
            ##ASSIGNING THE COORDINATES OF REQUIRED PARTS
            dist_list = []
               
            '''
            #if(cocoDict[CocoPart.Neck])
                dNeck = distance(cocoDict[CocoPart.Neck], cocoDict[CocoPart.LWrist]))
                dist_list.append('d_Neck', d_Neck)
            print(dist_list)
            '''
            #USING SYNTAX OF : https://pythonprogramming.net/zip-intermediate-python-tutorial/
            #part_list = ['r_wrist','l_wrist','l_ankle','r_ankle','l_elbow','r_elbow','l_hip','r_hip','l_shoulder','r_shoulder']
           
            '''
            print('-------------------------------')
            for name, member in CocoPart.__members__.items():
                    print(name, member.value)
            #print(CocoPart['REar'])
            '''
            ##print('HERE IS THE AVAILABLE PARTS LIST...')
            #print(cocoBind)
            ##print('BINDING THESE PARTS WITH THE COORDINATES AVAILABLE')
            
            cocoDict = dict(zip(cocoBind,zip(xlist,ylist)))

            #TRYING TO LOCATE PART NEAR THE HEAD FOR REFERENCE 
            try:
                RefPt = cocoDict[CocoPart.Nose]
            except KeyError:
                print('NOSE not detected...trying NECK')
                try: 
                    RefPt = cocoDict[CocoPart.Neck]
                except KeyError:
                        print('NECK not detected...trying LEar')
                        try: 
                            RefPt = cocoDict[CocoPart.LEar]
                        except KeyError:
                                print('LEar not detected...trying REar')
                                try: 
                                    RefPt = cocoDict[CocoPart.REar]
                                except KeyError:
                                    print('REar not detected...trying LEye')
                                    try: 
                                        RefPt = cocoDict[CocoPart.LEye]
                                    except KeyError:
                                            print('REar not detected...trying LEye')
                                            try: 
                                                RefPt = cocoDict[CocoPart.LEye]
                                            except KeyError:
                                                print('No PART detected')

                        

            #PROBABLY UNABLE TO DETECT THE MISSING KEY. HENCE USING THE METHOD DESCRIBED HERE:
            #https://realpython.com/python-keyerror/

            try:
                d_RShoulder = distance(RefPt, cocoDict[CocoPart.RShoulder])
                dist_dict.update({'RShoulder': d_RShoulder})
            except KeyError:
                pass
                ##print('Not There') 

            try:
                d_RElbow = distance(RefPt, cocoDict[CocoPart.RElbow])
                dist_dict.update({'RElbow': d_RElbow})
            except KeyError:
                pass
                ##print('Not There') 

            try:
                d_RWrist = distance(RefPt, cocoDict[CocoPart.RWrist])
                dist_dict.update({'RWrist': d_RWrist})
            except KeyError:
                pass
                ##print('Not There')

            try:
                d_LShoulder = distance(RefPt, cocoDict[CocoPart.LShoulder])
                dist_dict.update({'LShoulder': d_LShoulder})
            except KeyError:
                pass
                ##print('Not There')  

            try:
                d_LElbow = distance(RefPt, cocoDict[CocoPart.LElbow])
                dist_dict.update({'LElbow': d_LElbow})
            except KeyError:
                pass
                ##print('Not There')  

            try:
                d_LWrist = distance(RefPt, cocoDict[CocoPart.LWrist])
                dist_dict.update({'LWrist': d_LWrist})
            except KeyError:
                pass
                ##print('Not There') 

            try:
                d_LHip = distance(RefPt, cocoDict[CocoPart.LHip])
                dist_dict.update({'LHip': d_LHip})
            except KeyError:
                pass
                ##print('Not There')

            try:
                d_RHip = distance(RefPt, cocoDict[CocoPart.RHip])
                dist_dict.update({'RHip': d_RHip})
            except KeyError:
                pass
                ##print('Not There')
            
            try:
                d_LKnee = distance(RefPt, cocoDict[CocoPart.LKnee])
                dist_dict.update({'LKnee': d_LKnee})
            except KeyError:
                pass
                ##print('Not There')

            try:
                d_RKnee = distance(RefPt, cocoDict[CocoPart.RKnee])
                dist_dict.update({'RKnee': d_RKnee})
            except KeyError:
                pass
                ##print('Not There')

            try:
                d_LAnkle = distance(RefPt, cocoDict[CocoPart.LAnkle])
                dist_dict.update({'LAnkle': d_LAnkle})
            except KeyError:
                pass
                ##print('Not There')

            try:
                d_RAnkle = distance(RefPt, cocoDict[CocoPart.RAnkle])
                dist_dict.update({'RAnkle': d_RAnkle})
            except KeyError:
                pass
                ##print('Not There')
               
            ##print(dist_dict)
            
            #GETTING KEY WITH MAX VALUE AND MIN VALUE
            #PRINTING THE FARTHEST AND NEAREST POINT WITH Nose AS THE ORIGIN (AKA, REFERENCE POINT)
            from operator import itemgetter
            try:
                far_part = max(dist_dict, key=dist_dict.get)
                print('FARTHEST : ',far_part)
            except ValueError:
                far_part = 'NONE'
                #print('INSUFFICIENT PARTS DETECTED ! Retrying...')
            try:
                near_part = min(dist_dict, key=dist_dict.get)
                print('CLOSEST : ',near_part)
            except ValueError:
                near_part = 'NONE'
                #print('INSUFFICIENT PARTS DETECTED ! Retrying...')
            #TRYING TO ESTIMATE THE POSTURE
            #--> RATIO OF LENGTH OF THE VISIBLE FEMUR BONE AND SHIN
            try:
                #RATIO FOR THE LEFT LEG
                ratio1 = distance(cocoDict[CocoPart.LKnee], cocoDict[CocoPart.LAnkle]) / distance(cocoDict[CocoPart.LKnee], cocoDict[CocoPart.LHip] ) 
                #print(ratio1)
            except KeyError:
                ratio1 = 0;     #BASED ON THE RECOMMENDATION HERE (https://stackoverflow.com/questions/56490222/unable-to-resolve-the-unboundlocalerror-in-python3-7)
                ##print('Left Ratio Not Available')    
            
            try:
                #RATIO FOR THE RIGHT LEG    
                ratio2 = distance(cocoDict[CocoPart.RKnee], cocoDict[CocoPart.RAnkle] ) / distance(cocoDict[CocoPart.RKnee], cocoDict[CocoPart.RHip] )
                #print(ratio2)
            except KeyError:
                ratio2 = 0
                ##print('Right Ratio Not Available')
            
            try:
                print('max ratio is : ', max(ratio1,ratio2))
            except KeyError:
                try:
                    print(ratio1)
                except KeyError:
                    try:
                        print(ratio2)
                    finally:
                        print('No Ratio Available.')
            #--> ANGLE OF INCLINATION OF THE FEMUR w.r.t SHIN
            try:
                left_m1 = slopeOf(cocoDict[CocoPart.LKnee], cocoDict[CocoPart.LAnkle])
            except KeyError:
                left_m1 = 0
            
            try:
                left_m2 = slopeOf(cocoDict[CocoPart.LKnee], cocoDict[CocoPart.LHip])
            except KeyError:
                left_m2 = 0
            
            left_theta = math.degrees(math.atan((left_m2-left_m1)/(1+left_m2*left_m1)))
            #print('left_theta',left_theta)
            try:
                right_m1 = slopeOf(cocoDict[CocoPart.RKnee], cocoDict[CocoPart.RAnkle])
            except KeyError:
                right_m1 = 0
            
            try:
                right_m2 = slopeOf(cocoDict[CocoPart.RKnee], cocoDict[CocoPart.RHip])
            except KeyError:
                right_m2 = 0
            
            right_theta = math.degrees(math.atan((right_m2-right_m1)/(1+right_m2*right_m1)))
            #print(right_theta)           
            ##plt1.plot(xlist,ylist,'ro')
            #INVERTING THE Y AXIS i.e, TOP TO BOTTOM : 0 TO MAX
            ##plt1.gca().invert_yaxis()
            #CHECK WHETHER THE AXIS IS INVERTED OR NOT
            ##print(plt1.gca().yaxis_inverted())
            #plt1.show()
            
            #ESTIMATION OF POSTURE
            SleepingOrNot_max = SleepingOrNot_max + 1
            #if (max(ratio1,ratio2) > 1.81  or left_theta < 65 or right_theta+90 < 65 ) :    #ADDED 90 AS IT SHOWED 90 OFFSET 
            if (max(ratio1,ratio2) > 1.81  or (left_theta < 65 and left_theta >0)) :
                #print('SITTING')
                pass
            elif (max(ratio1,ratio2) == 0) :
                #print('ESTIMATOR :     LYING') 
                SleepingOrNot = SleepingOrNot + 1
            elif (far_part =='LAnkle' or far_part == 'RAnkle') :
                #print('STANDING')
                pass
            else :
                pass
            ###END

            # draw line
            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue

                # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
                cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

             #plt1.gca().invert_yaxis() 
            #UNCOMMENT TO PRINT ALL THE PARTS DETECTED AND THEIR COORDINATES MEASURED
            #print('COCO DICT: ', cocoDict)
            #print('DIST DICT: ', dist_dict) 
                
        #POPO EDIT  
        #try: 
        #return npimg               
        return npimg, SleepingOrNot, cocoDict
        #except:
        
        ##END

    def _get_scaled_img(self, npimg, scale):
        get_base_scale = lambda s, w, h: max(self.target_size[0] / float(h), self.target_size[1] / float(w)) * s
        img_h, img_w = npimg.shape[:2]

        if scale is None:
            if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
                # resize
                npimg = cv2.resize(npimg, self.target_size, interpolation=cv2.INTER_CUBIC)
            return [npimg], [(0.0, 0.0, 1.0, 1.0)]
        elif isinstance(scale, float):
            # scaling with center crop
            base_scale = get_base_scale(scale, img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)

            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros(
                    (max(self.target_size[1], npimg.shape[0]), max(self.target_size[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1], 0.2)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 2:
            # scaling with sliding window : (scale, step)
            base_scale = get_base_scale(scale[0], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros(
                    (max(self.target_size[1], npimg.shape[0]), max(self.target_size[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            window_step = scale[1]

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1],
                                  window_step)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 3:
            # scaling with ROI : (want_x, want_y, scale_ratio)
            base_scale = get_base_scale(scale[2], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            ratio_w = self.target_size[0] / float(npimg.shape[1])
            ratio_h = self.target_size[1] / float(npimg.shape[0])

            want_x, want_y = scale[:2]
            ratio_x = want_x - ratio_w / 2.
            ratio_y = want_y - ratio_h / 2.
            ratio_x = max(ratio_x, 0.0)
            ratio_y = max(ratio_y, 0.0)
            if ratio_x + ratio_w > 1.0:
                ratio_x = 1. - ratio_w
            if ratio_y + ratio_h > 1.0:
                ratio_y = 1. - ratio_h

            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, ratio_w, ratio_h)]

    def _crop_roi(self, npimg, ratio_x, ratio_y):
        target_w, target_h = self.target_size
        h, w = npimg.shape[:2]
        x = max(int(w * ratio_x - .5), 0)
        y = max(int(h * ratio_y - .5), 0)
        cropped = npimg[y:y + target_h, x:x + target_w]

        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w < target_w or cropped_h < target_h:
            npblank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

            copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
            npblank[copy_y:copy_y + cropped_h, copy_x:copy_x + cropped_w] = cropped
        else:
            return cropped

    def inference(self, npimg, resize_to_default=True, upsample_size=1.0):
        if npimg is None:
            raise Exception('The image is not valid. Please check your image exists.')

        if resize_to_default:
            upsample_size = [int(self.target_size[1] / 8 * upsample_size), int(self.target_size[0] / 8 * upsample_size)]
        else:
            upsample_size = [int(npimg.shape[0] / 8 * upsample_size), int(npimg.shape[1] / 8 * upsample_size)]

        if self.tensor_image.dtype == tf.quint8:
            # quantize input image
            npimg = TfPoseEstimator._quantize_img(npimg)
            pass

        logger.debug('inference+ original shape=%dx%d' % (npimg.shape[1], npimg.shape[0]))
        img = npimg
        if resize_to_default:
            img = self._get_scaled_img(npimg, None)[0][0]
        peaks, heatMat_up, pafMat_up = self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up], feed_dict={
                self.tensor_image: [img], self.upsample_size: upsample_size
            })
        peaks = peaks[0]
        self.heatMat = heatMat_up[0]
        self.pafMat = pafMat_up[0]
        logger.debug('inference- heatMat=%dx%d pafMat=%dx%d' % (
            self.heatMat.shape[1], self.heatMat.shape[0], self.pafMat.shape[1], self.pafMat.shape[0]))

        t = time.time()
        humans = PoseEstimator.estimate_paf(peaks, self.heatMat, self.pafMat)
        logger.debug('estimate time=%.5f' % (time.time() - t))
        return humans


if __name__ == '__main__':
    import pickle

    f = open('./etcs/heatpaf1.pkl', 'rb')
    data = pickle.load(f)
    logger.info('size={}'.format(data['heatMat'].shape))
    f.close()

    t = time.time()
    humans = PoseEstimator.estimate_paf(data['peaks'], data['heatMat'], data['pafMat'])
    dt = time.time() - t;
    t = time.time()
    logger.info('elapsed #humans=%d time=%.8f' % (len(humans), dt))
