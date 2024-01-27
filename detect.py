import cv2
from ultralytics import YOLO

class Detect:

    def __init__(self , classes , input_video , roi) -> None:
        self.model = self.load_model()
        self.classes = classes
        self.roi = roi
        self.input_video = input_video

    def load_model(self):
        model = YOLO('yolov8n.pt')
        model.fuse()
        return model
    
    def detect(self):
        return self.model(self.input_video , classes = 0)

    def draw_bb(self , frame , fins , f_num):
        for xyxy in fins[f_num]:
            frame = cv2.rectangle(frame , (int(xyxy[0]) , int(xyxy[1])) , (int(xyxy[2]) , int(xyxy[3])) , (255 , 0 , 0) , 3)

        return frame
    
    def plotbb(self ):
        results = self.detect()

        xyxys = []

        for result in results:
            boxes = result.boxes.cpu().numpy()

            xyxys.append(boxes.xyxy)

        fins = []
        for f in xyxys:
            inner_fin = []
            for xyxy in f:
                x = (xyxy[0] + xyxy[2])/ 2
                y = (xyxy[1] + xyxy[3]) / 2
                if ((x >= self.roi[0]) & (x <= self.roi[0] + self.roi[2]) & (y >= self.roi[1]) & (y <= self.roi[1] + self.roi[3])):
                    inner_fin.append(xyxy)
            fins.append(inner_fin)

        video = cv2.VideoCapture(self.input_video)
        
        f_num = 0
        while True:
            ret , frame = video.read()

            if not ret:
                break

            frame = self.draw_bb(frame = frame , fins= fins , f_num = f_num )
            f_num +=1
            cv2.imshow('Bounding Boxes', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

        return fins