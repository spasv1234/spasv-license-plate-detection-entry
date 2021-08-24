import jetson.utils
#Modules for working with database
import database

#Modules for working with license plate regex and email
import lp_filter,lp_alert

#Modules for working with time
import time
from datetime import datetime
import schedule

#Modules for working with paths
import os.path

#Modules for working with AI and images
from PIL import Image
import cv2
import numpy as np
import easyocr

#Config file
import config

#Loads object detection model
import jetson.inference
net = jetson.inference.detectNet(argv=['--model=../python/training/detection/ssd/models/license_plate_512_2/ssd-mobilenet.onnx','--labels=../python/training/detection/ssd/models/license_plate_512_2/labels.txt',
                                       '--input-blob=input_0','--output-cvg=scores','--output-bbox=boxes','--threshold=0.5'])
#Loads OCR model
reader = easyocr.Reader(['en'],gpu=True) 

#Video IO settings
camera = jetson.utils.videoSource(config.video_source,["--input-width=1280","--input-height=720","--input-loop=-1"]) 
#camera = jetson.utils.videoSource(config.video_source,["--input-width=1280","--input-height=720"]) 
#camera = jetson.utils.videoSource("/dev/video1")      # '/dev/video0' for V4L2
#display = jetson.utils.videoOutput() # 'my_video.mp4' for file  
#camera = jetson.utils.gstCamera(1920, 1080, "1")
#display = jetson.utils.glDisplay()

jetson.utils.cudaDeviceSynchronize()

bgr_img=None
license_plate_number_dict={}
authorised_license_plate_number=[]
temp_dict={}
location_id=2

#Function to crop image
def crop(img,x,y,w,h):
        crop_roi = (x, y, x+w, y+h)
        crop_img = jetson.utils.cudaAllocMapped(width=w,height=h,format=img.format)
        jetson.utils.cudaDeviceSynchronize()
        try:
                jetson.utils.cudaCrop(img, crop_img, crop_roi)
                return crop_img
        except:
                print("error cropping image")
                return img

#Function to convert cudaImage(rgb format) to bgr format for opencv
def convert_to_cv_img(rgb_img,bgr_img):
        if bgr_img is None:
                bgr_img = jetson.utils.cudaAllocMapped(width=rgb_img.width,height=rgb_img.height,format='bgr8')
        jetson.utils.cudaConvertColor(rgb_img, bgr_img)
        jetson.utils.cudaDeviceSynchronize()
        cv_img = jetson.utils.cudaToNumpy(bgr_img)
        jetson.utils.cudaDeviceSynchronize()
        return cv_img

#Function to load authorised vehicle list
def load_authorised_vehicle_list(file_name = "authorised_vehicle_list.txt" ):
        authorised_vehicle_list = []
        if not os.path.exists(file_name):
                f = open(file_name,"a")
                f.close()
        if os.path.exists(file_name):
                f=open(file_name,"r")
                lines = f.readlines()
                for line in lines:
                        authorised_vehicle_list.append(line.strip())
        return authorised_vehicle_list
                        
        
#Function to write to csv
def write_to_csv(license_plate_number,first_detected,last_detected,file_name = "vehicle_record.txt"):
        if not os.path.exists(file_name):
                f = open(file_name,"w")
                f.write("License Plate Number,First Detected,Last Detected")
                f.close()
        if os.path.exists(file_name):
                f=open(file_name,"a")
                f.write("\n"+license_plate_number+","+first_detected+","+last_detected)
                f.close()
                

#Function to start detection
def detect_license_plate():
        license_plate_number_list=[]
        license_plate_number_list.clear()                                                                                                                                       #Ensure that license_plate_number_list is cleared from previous operations
        authorised_vehicle_list = load_authorised_vehicle_list()
        
        #Loop timing control
        loop_start_time = time.time()                                                                                                                                           #Current time of loop                                                                                                                                                    #Duration of loop in seconds
        
        while True:
                img = camera.Capture()
                cv_img=None
                detections = net.Detect(img,overlay='none')                                                                                                                     #overlay set to none to disable detectnet bounding boxes

                jetson.utils.cudaDeviceSynchronize()
                #Checks for detection
                if len(detections) > 0:
                        for detection in detections:
                                #Cropping license plate
                                x,y,w,h = round(detection.Left),round(detection.Top),round(detection.Width),round(detection.Height)                                             #Gets information for cropping and drawing boxes
                                cv2.rectangle(jetson.utils.cudaToNumpy(img),(x,y),(x+w,y+h),color=(0,255,0),thickness=2)                                                        #Draws bounding box
                                cropped_img = crop(img,x,y,w,h)                                                                                                                 #Crops image

                                #OCR
                                jetson.utils.cudaDeviceSynchronize()
                                results = reader.readtext(jetson.utils.cudaToNumpy(cropped_img), detail=0)                                                                      #Reads text with ocr
                                license_plate_number = ''                                                                                                                       #Variable to form license plate

                                for result in results:
                                        result = result.upper()                                                                                                                 #Convert to uppercase
                                        license_plate_number += result                                                                                                          #Concatenate multi lines license plate into 1 if needed
                                        license_plate_number = lp_filter.remove_noise(license_plate_number)                                                                     #Removes borders of license plate if detected
                                        #License plate detected
                                        if (lp_filter.validate_license_plate_with_checksum(license_plate_number)):
                                                #Have not been processed in this detection loop
                                                if license_plate_number not in license_plate_number_list:
                                                        #If vehicle has not visited at an earlier time
                                                        if license_plate_number not in license_plate_number_dict:
                                                                #Authorised parking only time
                                                                if ((time.localtime().tm_hour >= config.authorized_parking_min_hour or config.authorized_parking_max_hour)
                                                                    and license_plate_number not in authorised_vehicle_list):
                                                                        cv_img = convert_to_cv_img(img,bgr_img)                                                                 #Convert to opencv img for saving
                                                                        cv2.imwrite("email_attachment.jpg",cv_img)                                                              #Saves img to be sent as attachment
                                                                        #lp_alert.send_lp_alert(license_plate_number,"email_attachment.jpg")                                     #Send email with attachment
                                                                        print("Sent Email")
                                                        license_plate_number_list.append(license_plate_number)                                                                  #Appends license plate number to list of already processed license plate so it won't be processed again
                                print("License Plate Detected: "+license_plate_number)
                else:
                        print("NO DETECTIONS")
                

                if config.enable_cv2_video_stream:
                        #Converts to opencv format
                        if cv_img is None:
                                cv_img = convert_to_cv_img(img,bgr_img)
                        cv2.putText(cv_img,"License Plate Detected:",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        for i in range(0,len(license_plate_number_list)):
                                cv2.putText(cv_img,license_plate_number_list[i],(10,60+(30*i)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                                
                        cv2.imshow("output",cv_img)                                                                                                                  #Display current detection frame
                        k = cv2.waitKey(1) & 0xFF
                        del img

                loop_current_time = time.time()                                                                                                                                 #Get current time of loop
                loop_elapsed_time = loop_current_time - loop_start_time                                                                                                         #Find out time that loop has gone on for
                if loop_elapsed_time > config.loop_duration:                                                                                                                    #If loop has ran for more than intended duration,break the loop
                        print(loop_elapsed_time)
                        break

        print(license_plate_number_list)
        #check for new license plate number
        for license_plate_number in license_plate_number_list:
                if config.enable_write_to_database:
                        database.add_vehicle_to_database(license_plate_number)
                        database.add_vehicle_entry_record(license_plate=license_plate_number,time_detected=datetime.now().replace(microsecond=0),location_id=1)

#This is the main program loop
#license plate detection is detected at the start of every hour
#for easier debugging and testing, the function can be ran directly without scheduling

#Main program loops, loops when webcam feed is active
#reader.readtext("email_attachment.jpg", detail=0)
while True:
        jetson.utils.cudaDeviceSynchronize()
        load_authorised_vehicle_list()
        detect_license_plate()
        
        
