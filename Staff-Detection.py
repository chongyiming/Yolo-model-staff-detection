from ultralytics import YOLO
import cv2
import cvzone
import os

#Get video path
cap = cv2.VideoCapture(
    "video.mp4"
)

#Initialize object detection model
model = YOLO('yolo26n.pt')
image_model = YOLO('best.pt')

#Initialize frame counter
currentframe=0

#Create data folder if there isn't data folder (store frame and xy coordinates)
if not os.path.exists('data'):
    os.makedirs('data')

#Get all x,y for each object detection in single frame
def process_box(box):
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    confidence = float('{0:.2f}'.format(box.conf[0]))
    return x1,y1,x2,y2,confidence



while True:
    success, img = cap.read()

    #If video end then break so it wont output error
    if not success:
        print("Video ended")
        break

    #Keeps results for current image in memory
    results = model(img, stream=True)

    #Loop the result of objects detection in current frame
    for r in results:
        #Get all the bounding boxes found in current frame.
        boxes=r.boxes
        #Every object will have own class number, in this case person class is 0 so we check for box.cls ==0
        for box in boxes:
            if int(box.cls) == 0:
                px1, py1, px2, py2, confidence=process_box(box)
                #Only select person detection with higher confidence level
                if confidence >0.3:
                    #Put a "box" with Person + confidence level in current frame
                    cvzone.putTextRect(img, f'Person {confidence}', (max(0, px1), max(35, py1)), scale=0.6,
                                       thickness=1, offset=3)
                    cvzone.cornerRect(img, (px1, py1, px2-px1, py2-py1), l=9)

                    #This is for nametag detection result
                    extracted_results = image_model(img, stream=True)
                    #Loop result for each frame
                    for i in extracted_results:
                        extracted_boxes=i.boxes
                        #Loop every object detection in single frame
                        for box in extracted_boxes:
                            #Since we only use one class to train Yolo model, so we only need to check for box.cls == 0, there will not be another object detected
                            if int(box.cls) == 0:
                                x1, y1, x2, y2,confidence = process_box(box)
                                if confidence > 0.3:
                                    #We assume staff will always wear name tag, which means x and y coordinates of nametag should be within Person x and y coordinates
                                    if x1 >= px1 and y1 >= py1 and x2 <= px2 and y2 <= py2:
                                        cvzone.putTextRect(img, f'Nametag {confidence}',
                                                           (max(0, x1), max(35, y1)), scale=0.6,
                                                           thickness=1, offset=3)
                                        cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), l=9)
                                        #Store the frame that detect person and nametag is within Person x and y coordinates
                                        cv2.imwrite('./data/frame' + str(currentframe) + '.jpg', img)
                                        #Store the person x and y coordinates
                                        file = open('./data/frame' + str(currentframe) + '.txt', "w")
                                        file.write(f'[{px1},{px2},{py1},{py2}]')
                                        file.close()
                                        currentframe+=1
                                        break

    #create window to show current frame
    cv2.imshow("Image",img)
    #1ms delay between frames to render image
    cv2.waitKey(1)