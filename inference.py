import cv2
import numpy as np
import onnxruntime as ort

class YOLOv8Detector:

    classes=['fire','head','helmet','person','smoke']

    colors={
        'fire':(0,0,255),       #red
        'head':(255,0,0),       #blue
        'helmet':(0,255,0),     #green
        'person':(0,0,0),       #white
        'smoke':(255,255,255)   #black
    }

    def __init__(self,onnx_model_path:str,conf_threshold:float=0.5,iou_threshold:float=0.45,use_gpu:bool=0):
        self.onnx_model_path=onnx_model_path
        self.conf_threshold=conf_threshold
        self.iou_threshold=iou_threshold
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

        #load model
        self.session= ort.InferenceSession(self.onnx_model_path,providers=providers)
        #get information of model's input
        self.input_name = self.session.get_inputs()[0].name
        input_shape=self.session.get_inputs()[0].shape
        self.input_height=input_shape[2]
        self.input_width=input_shape[3]

    def preprocess(self, image:np.ndarray):
        orig_h,orig_w=image.shape[:2]
        resized=cv2.resize(image,(self.input_width,self.input_height))
        rgb=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
        normalized=rgb/255.0
        input_tensor=np.transpose(normalized,(2,0,1))
        input_tensor=np.expand_dims(input_tensor,axis=0)
        return input_tensor.astype(np.float32),orig_h,orig_w

    def postprocess(self,outputs:np.ndarray,orig_h:int,orig_w:int):
        outputs=outputs[0]
        boxes=outputs[...,:4]
        confidences=outputs[...,4]
        class_scores=outputs[...,5:]
        class_ids=np.argmax(class_scores,axis=1)
        #best category
        scores=confidences*np.max(class_scores,axis=1)
        #filter low confidence result
        keep_indices=scores>self.conf_threshold
        boxes=boxes[keep_indices]
        scores=scores[keep_indices]
        class_ids=class_ids[keep_indices]
        if len(boxes)==0:
            return []

        #carry out NMS,remove the duplicate boxes
        indices=self.nms(boxes,scores)
        boxes=boxes[indices]
        scores=scores[indices]
        class_ids=class_ids[indices]

        result=[]
        for box,score,class_id in zip(boxes,scores,class_ids):
            x_center,y_center,width,height=box
            scale_x=orig_w/self.input_width
            scale_y=orig_h/self.input_height

            x1=(x_center-width/2)*scale_x
            y1=(y_center-height/2)*scale_y
            x2=(x_center+width/2)*scale_x
            y2=(y_center+height/2)*scale_y

            x1=max(0,min(x1,orig_w))
            y1=max(0,min(y1,orig_h))
            x2=max(0,min(x2,orig_w))
            y2=max(0,min(y2,orig_h))

            result.append({
                'class':self.classes[class_id],
                'score':float(score),
                'box':[int(x1),int(y1),int(x2),int(y2)],
            })
        return result
    def nms(self, boxes:np.ndarray,scores:np.ndarray):
        #box array shape:(N,4)
        #confidece array shape:(N,)
        x1=boxes[:,0]-boxes[:,2]/2
        y1=boxes[:,1]-boxes[:,3]/2
        x2=boxes[:,0]+boxes[:,2]/2
        y2=boxes[:,1]+boxes[:,3]/2

        areas=(x2-x1+1)*(y2-y1+1)
        order=scores.argsort()[::-1]
        keep=[]
        while order.size > 0:
            i=order[0]
            keep.append(i)
            xx1=np.maximum(x1[i],x1[order[1:]])
            yy1=np.maximum(y1[i],y1[order[1:]])
            xx2=np.minimum(x2[i],x2[order[1:]])
            yy2=np.minimum(y2[i],y2[order[1:]])
            w=np.maximum(0,xx2-xx1+1)
            h=np.maximum(0,yy2-yy1+1)
            inter=w*h
            iou=inter/(areas[i]+areas[order[1:]]-inter)
            inds=np.where(iou<=self.iou_threshold)[0]
            order=order[inds+1]
        return np.array(keep)
    def detect(self,image:np.ndarray):
        input_tensor,orig_h,orig_w=self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        result=self.postprocess(outputs,orig_h,orig_w)
        return result
    def draw(self,image:np.ndarray,detections:list[dict]):
        img_with_detection=image.copy()
        for detection in detections:
            class_name=detection['class']
            score=detection['score']
            x1,y1,x2,y2=detection['box']
            color=self.colors[class_name]
            cv2.rectangle(img_with_detection,(x1,y1),(x2,y2), color, 2)
            label=f"{class_name}:{score:.2f}"
            cv2.putText(img_with_detection,label,(x1,y1-10),cv2.FONT_HERSHEY_PLAIN,0.5,color,2)
        return img_with_detection

if __name__=='__main__':
    Onnx_model_path=r'D:\yang\Downloads\ultralytics-main\ultralytics-main\runs\detect\train2\weights\best.onnx'
    Test_image_path='test.jpg'
    Output_image_path='result.jpg'
    Conf_threshold=0.4
    Iou_threshold=0.45 #nms iou
    Use_gpu=False

    #initial detector
    detector = YOLOv8Detector(
        onnx_model_path=Onnx_model_path,
        conf_threshold=Conf_threshold,
        iou_threshold=Iou_threshold,
        use_gpu=Use_gpu
    )

    image=cv2.imread(Test_image_path)
    if image is None:
        raise FileNotFoundError(f'cannot find {Test_image_path}')
    print('testing in progress\n')
    detections=detector.detect(image)
    print('result:')
    for i,det in enumerate(detections,1):
        print(f"{i}.category:{det['class']},confidence:{det['score']:.2f},location:{det['box']}")
    result_image=detector.draw(image,detections)
    cv2.imshow('result',result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(Output_image_path,result_image)
    print(f'saved {Output_image_path}')