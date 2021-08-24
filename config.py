#Video Feed Settings
#"rtsp://172.23.32.123:8080/h264_ulaw.sdp" #"Test_Videos/test_buggy2.mp4" #"/dev/video1" for webcam
video_source = "Test_Videos/demo_video.mp4"

#Detection loop settings
loop_duration = 60 #In seconds

#Email settings
#requires less secure app to be turned on in google 
email_login_username = "spasv1234"  
email_login_password = "P@ssword1234"

#Database settings
enable_write_to_csv = True        #True to write to csv, false to ignore 
enable_write_to_database = True   #True to write to mysql database, false to ignore, requires same network
database_connection = 'mysql+pymysql://root:Password1234@192.168.1.100:3306/webapp_database'

#Authorized parking setting
authorized_parking_min_hour = 11
authorized_parking_max_hour = 5

#CV2 video stream options
enable_cv2_video_stream = True #True to view detection live, useful for debugging but resource intensive
