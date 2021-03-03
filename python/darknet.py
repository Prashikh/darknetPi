from ctypes import *
import math
import random
import argparse

import os
import os.path as osp

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#TODO: define a root level folder
class DETECT():
    def __init__(self, path, cfg, weights, data):

        # set defaults
        self.dllpath = path
        self.cfg = cfg
        self.weights = weights
        self.data = data

        # change working directory to ensure nothing breaks
        self.setroot(self.dllpath) 

        # load functions from dll
        self.lib = CDLL(self.dllpath, RTLD_GLOBAL)
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.predict = self.lib.network_predict
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.set_gpu = self.lib.cuda_set_device
        self.set_gpu.argtypes = [c_int]

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = self.lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.lib.network_predict
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.detected_objects = None

        self.setup()
    
    def setroot(self, path):
        paths = osp.split(path)
        if paths[0]:
            os.chdir(paths[0])

    def setup(self):
        # make sure cfg, weights, and data are file paths that are byte strings
        self.net = self.load_net(self.cfg, self.weights, 0)
        self.meta = self.load_meta(self.data)


    def classify(self, im):
        out = self.predict_image(self.net, im)
        res = []
        for i in range(self.meta.classes):
            res.append((self.meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def detect(self, image, thresh=.5, hier_thresh=.5, nms=.45):
        im = self.load_image(image, 0, 0)
        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(self.net, im)
        dets = self.get_network_boxes(self.net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms): self.do_nms_obj(dets, num, self.meta.classes, nms);

        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((self.meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        self.detected_objects = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_detections(dets, num)

    def output(self, objects, thresh, print_dections):
        
        amounts = dict.fromkeys(objects,0)
        for obj in self.detected_objects:
            if print_dections:
                print(f"Object is a {obj[0].decode('ASCII')} with probability {(obj[1]*100):.2f}%")

            if obj[0].decode('ASCII') in objects and obj[1] >= thresh:
                amounts[obj[0].decode('ASCII')] += 1
        
        return amounts

    def parse_image(self, image, objects=None, print_detections=False):
        self.detect(image)
        objects = self.output(objects, 0.8, print_detections)
 
        if print_detections:       
            for obj in objects:
                print(f'Detected {objects[obj]} {obj}(s) in the image')
            print('----------------------------------------------------------------------------\n\n')

        return objects


def parse_args():
    # arg parser
    # TODO: maybe add args to specify weights, data, and cfg files
    # TODO: for files maybe only specify the name and use os.walk to see if they exist in the repo somewhere
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file-path",action="store",
                        help="point to the dll file location for the darknet executable file - default value: ./libdarknet.so",
                        dest="filepath",type=str, default="./libdarknet.so")
    parser.add_argument("-w","--weights",action="store",
                        help="point to the weights file location for the classifier - default value: ./yolov3.weights",
                        dest="weights",type=str,default="./yolov3.weights")
    parser.add_argument("-d","--debug",action="store_true",
                        help="indicate if debug prints should be printed in the terminal",
                        dest="debug")

    args = parser.parse_args()

    return args

    
if __name__ == "__main__":
    
    # initialize the classifier
    args = parse_args()
    classifier = DETECT(args.filepath, b"cfg/yolov3.cfg", args.weights, b"cfg/coco.data")
    objects = ["car", "truck", "person"]
    classifier.parse_image(b"data/cars.jpg", objects, args.debug)

    # detect the objects in the specified images



    

