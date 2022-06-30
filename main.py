from glob import glob
import os.path
import ui
import definitions as d
import face_recognition as fr
import cv2
import pickle
import numpy as np
import winsound
import random

img_dir = "photo"

print("Face Attendance System")

print("\n- The Shire -")
print("Sintya Tri Wahyu Adityawati (0706012010012)")
print("Timothyus Kevin Dewanto (0706012010040)")
print("David Alexander (0706012010043)")
print("Armaida Gholia Lestari (0706012010060)")

print("\nStarting program...")

def camera(list_folder):
    ## Camera / Face Detection Function
    
    # Constants needed
    # Constant is kept out of cam is opened,
    # Because if it's inside, it can't update
    frame_name = None
    attendence_message = None
    
    eye_blink = 0
    blink_total = 0
    blink_number = 0
    
    session_time = 0
    
    # List folder
    names = list(map(d.get_names, list_folder))
    
    # Load face encodings
    face_encode = np.load('assets/face_encodings.npy')
    
    print("\nInitiating camera...\n")
    
    # Turn on the camera
    cam = cv2.VideoCapture(0)
    
    # While camera is on,
    while cam.isOpened():
        # Capture camera frame by frame
        ret, frame = cam.read()
        
        # Get face locations, landmarks and encodings
        frame_face_loc = fr.face_locations(frame)
        frame_face_landmarks = fr.face_landmarks(frame, frame_face_loc)
        frame_face_encode = fr.face_encodings(frame, frame_face_loc)
        
        # Turn on the camera and identify face in frame
        for index, (loc, encode, landmark) in enumerate(zip(frame_face_loc, frame_face_encode, frame_face_landmarks)):
            
            # Find matching faces
            score = fr.face_distance(face_encode, encode)
            index_match = np.argmin(score)
            
            # Face recognition threshold is 0.5
            # If min(score) is more than the threshold,
            # Face is unknown or not detected
            if np.min(score) < 0.5:
                # For comparison matching
                temp = frame_name
                
                # Frame name matching
                frame_name = names[index_match]
            else:
                frame_name = "Unknown"
                session_time = 0
            
            # Recording attendance
            # If face is unknown, don't track the eyes
            # and record attendance
            if not frame_name == "Unknown":
                # Else, Track the eyes and record the attendance
                
                session_time = session_time + 1
                
                print(session_time)
                
                # Eyes detection
                left_eye = np.array(landmark['left_eye'], dtype=np.int32)
                right_eye = np.array(landmark['right_eye'], dtype=np.int32)
                
                # EAR = Eye Aspect Ratio
                # Get EAR ratio of both eyes
                EAR_avg = (d.get_EAR_ratio(left_eye) + d.get_EAR_ratio(right_eye)) / 2
                
                print(f"EAR avg: {EAR_avg}")
                
                # EAR ratio threshold = 0.25
                # Minimum frames of closed eyes = 2
                # Detect blinking of both eyes
                print(f"Eye blink: {eye_blink}")
                print(f"Blink total: {blink_total}")
                if EAR_avg < 0.25:
                    eye_blink += 1
                else:
                    if eye_blink >= 2:
                        blink_total += 1
                        
                        # Sound beep everytime blink total is recorded
                        winsound.Beep(2500, 500)
                        session_time = session_time - 2
                        
                        attendance_message = ""
                    
                    # Reset
                    eye_blink = 0
                    
                if session_time > 25:
                    # Reset
                    session_time = 0
                    eye_blink = 0
                    blink_total = 0
                    
                    attendance_message = "Session time out, please try again"
                
                if temp != frame_name:
                    # Reset eye blink total
                    blink_total = 0
                    
                    # Set random blink numbers
                    blink_number = random.randint(3, 4)
                    
                    attendence_message = ""
                
                if blink_number == blink_total:
                    # Reset
                    blink_number = 0
                    eye_blink = 0
                    
                    attendence_message = "Attendance is recorded"
                    
                    # Save to CSV file
                    d.csv_write(frame_name)
                    
                # Message
                blink_message = f"Blink {blink_number} times | Blink total: {blink_total}"
                
                # Draw eyes point and blink messages
                cv2.polylines(frame, [left_eye], True, (255,0,0) , 1)
                cv2.polylines(frame, [right_eye], True, (255,0,0) , 1)
                cv2.putText(frame,blink_message,(10,50),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,0),2)
                cv2.putText(frame,attendence_message,(20,450),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
                
            # Show display output
            cv2.rectangle(frame,(loc[3],loc[0]),(loc[1],loc[2]), (0,0,0),2) # top-right, bottom-left
            cv2.putText(frame,frame_name,(loc[3],loc[0]-3),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
        
        # Show display output
        cv2.imshow("Attendance System (Q - Quit, T - Go to UI)", frame)
        
		# Press 'q' is quit | Press 't' for UI
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quiting...")
            cam.release()
            cv2.destroyAllWindows()
            break
                
        if cv2.waitKey(1) & 0xFF == ord('t'):
            print("Going to UI...")
            cam.release()
            cv2.destroyAllWindows()
            ui.init()
                
    # End of camera function

# Read all images in 'people' directory
img_list = glob(img_dir + '/*.*')
print(f"Number of files in '{img_dir}' directory : {len(img_list)}")

if len(img_list) == 0:
    # Open up UI to make new image
    # Encode photo
    print("No photos found! Opening UI...")
    print("Restart after making the photos!")
    ui.init()
    
else:
    # Check Attendance CSV
    d.csv_check()
    
    # Load data from pickle file (n_people)
    if(os.path.exists('assets/n_people.pk') == False):
        print("Pickle file is not yet generated!")
        print("Generating face encoder and pickle file...")
        d.encode(img_list)
        
        # open up camera
        camera(img_list)
        
    else:
        with open('assets/n_people.pk', 'rb') as pickle_file:
            n_people_in_pickle = pickle.load(pickle_file)
        print(f"Number of files that should be in '{img_dir}' directory : {n_people_in_pickle}")
        
        if n_people_in_pickle == len(img_list):
            # open up camera
            camera(img_list)
            
        else:
            # reencode
            print("New photos has been added! Reencoding...")
            d.encode(img_list)
            
            # open up camera
            camera(img_list)
    
    
    
    
    
