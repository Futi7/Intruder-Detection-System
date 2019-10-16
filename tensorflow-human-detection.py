import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import smtplib, ssl
from timeit import default_timer as timer
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

sys.path.append("..")

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    model_path = 'coco/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    cap = cv2.VideoCapture(0)



    port = 465  
    password = 'Your_Password_Here'#Enter your mail password

    context = ssl.create_default_context()
    msg = MIMEMultipart()

    message = 'Intruder Detected !!'

    receiver_email = 'exampleReceiver@gmail.com'
    sender_email = "exampleSender@gmail.com"

    msg['From'] = "exampleSender@gmail.com"
    msg['To'] = 'exampleReceiver@gmail.com'
    msg['Subject'] = "Seventh Security System Alert"
 
    msg.attach(MIMEText(message, 'plain'))


    control = 0 

    while True:
        r, img = cap.read()
        img = cv2.resize(img, (400, 400))

        boxes, scores, classes, num = odapi.processFrame(img)

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                if control == 0:
                    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
                        cv2.imwrite('test.jpg',img)
                        img_data = open('test.jpg', 'rb').read()
                        msg.attach(MIMEImage(img_data))
                        server.login(sender_email, password)
                        server.sendmail(msg['From'], msg['To'], msg.as_string())
                        control += 1
                        start = timer()
                                    

                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
        end = timer()
        print("End", end)
        if (end - start >= 30 ):
            control = 0

        cv2.imshow("Security System", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

