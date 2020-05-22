import glob
import os
import cv2
import pickle
import torch

import _init_paths
import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
from tracking_utils.timer import Timer
import datasets.dataset.jde as datasets
from track import eval_seq
from tracker.multitracker import JDETracker

logger.setLevel(logging.INFO)

class ScoringService(object):
    @classmethod
    def get_model(cls, model_path='../model'):
        """Get model method
 
        Args:
            model_path (str): Path to the trained model directory.
 
        Returns:
            bool: The return value. True for success, False otherwise.
        
        Note:
            - You cannot connect to external network during the prediction,
              so do not include such process as using urllib.request.
 
        """
        model_files = glob.glob(model_path+'/*.pth')
        
        try:
            if len(model_files):
                cls.opt = opts().init(['mot','--load_model',model_files[0],'--arch','hrnet_18','--reid_dim','128','--conf_thres','0.9','--gpus','0'])
                cls.tracker = JDETracker(cls.opt, frame_rate=5)
            else:
                cls.opt = opts().init(['mot'])
                cls.tracker = None
            return True
        except:
            return False

        '''
        try:
            if len(model_files):
                model_file = model_files[0]
                with open(os.path.join(model_path, model_file), 'rb') as f:
                    cls.model = pickle.load(f)
            else:
                cls.model = None
            
            return True
        
        except:
            return False
        '''

    @classmethod
    def predict(cls, input):
        """Predict method
 
        Args:
            input (str): path to the video file you want to make inference from
 
        Returns:
            dict: Inference for the given input.
                format:
                    - filename []:
                        - category_1 []:
                            - id: int
                            - box2d: [left, top, right, bottom]
                        ...
        Notes:
            - The categories for testing are "Car" and "Pedestrian".
              Do not include other categories in the prediction you will make.
            - If you do not want to make any prediction in some frames,
              just write "prediction = {}" in the prediction of the frame in the sequence(in line 65 or line 67).
        """

        opt = cls.opt
        tracker = cls.tracker
        dataloader = datasets.LoadVideo(input, opt.img_size)
        
        predictions = []
        cap = cv2.VideoCapture(input)
        f_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        f_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        img_width = opt.img_size[0]
        img_height = opt.img_size[1]
        fname = os.path.basename(input)
        timer = Timer()
        #results = []
        frame_id = 0
        for path, img, img0 in dataloader:
            if tracker is not None:
                if frame_id % 20 == 0:
                    print(img.shape)
                    logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
                # run tracking
                timer.tic()
                if opt.gpus[0] >=0:
                    blob = torch.from_numpy(img).cuda().unsqueeze(0)
                else:
                    blob = torch.from_numpy(img).unsqueeze(0)
                online_targets = tracker.update(blob, img0)
                online_tlwhs = []
                online_ids = []
                online_classid = []
                prediction = {"Car":[], "Pedestrian":[]}
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    classid = t.classid
                    if tlwh[2] * tlwh[3] > opt.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_classid.append(classid)
                    #print(tlwh, tid, classid)
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    #box = {"id":tid ,"box2d":[x1*(f_width/img_width), y1*(f_height/img_height), x2*(f_width/img_width), y2*(f_height/img_height)]}
                    box = {"id":tid ,"box2d":[x1, y1, x2, y2]}
                    if int(classid) == 1:
                        prediction["Car"].append(box)
                    elif int(classid) == 0:
                        prediction["Pedestrian"].append(box)
            else:
                prediction = {"Car": [{"id": 0, "box2d": [0, 0, f_width, f_height]}],
                              "Pedestrian": [{"id": 0, "box2d": [0, 0, f_width, f_height]}]}

            timer.toc()
            print(frame_id)
            if frame_id == 1:
                print(prediction)
            frame_id += 1
            predictions.append(prediction)
        '''
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if cls.model is not None:
                prediction = cls.model.predict(frame)
            else:
                prediction = {"Car": [{"id": 0, "box2d": [0, 0, frame.shape[1], frame.shape[0]]}],
                              "Pedestrian": [{"id": 0, "box2d": [0, 0, frame.shape[1], frame.shape[0]]}]}
            predictions.append(prediction)
        cap.release()
        '''
        with open('pred.json', 'w') as file:
            file.write(json.dumps({fname: predictions}))

        return {fname: predictions}
