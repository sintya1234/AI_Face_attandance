from tqdm import tqdm
from glob import glob
import face_recognition as fr
import numpy as np
import cv2
import pickle
import os
import string
from scipy.spatial import distance as dist
import pandas as pd
from datetime import datetime

def csv_check():
    # Checks the Attendance CSV file.
    
    print(f"\nChecking Attendance CSV file...")
    if(os.path.exists('assets/attendance.csv') == False):
        # If file does not exist, make new CSV
        print("Attendance CSV does not exist!")
        print("Creating...")
        # create new pandas DataFrame
        # Column: Name, Date, Time
        df = pd.DataFrame(list(),
            columns=['Name', 'Day', 'Month-Day', 'Year', 'Time']
        )

        # write DataFrame to new csv file
        df.to_csv('assets/attendance.csv')
        
        print("Attendance CSV is made.")
    else:
        print("Attendance CSV file exists.")

def csv_write(name):
    with open('assets/attendance.csv', 'r+') as file:
        # Read lines in csv file, except first line
        lines = file.read().splitlines()[1:]
        
        # Store only names
        names = list(map(lambda line : line.split(',')[0], lines))
        
        if not name in names:
            # Create datetime object
            now = datetime.now()
            date = now.strftime("%A, %B %d, %Y")
            time = now.strftime("%H:%M:%S %p")
            file.writelines(f", {name},{date},{time}\n")
            print("Written")

def get_names(path):
    # Get name from photo folder
    name = path.split(os.sep)[-1].split('.')[0]
    return name

def get_images(path):
    # Get images from photo folder
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img
    
def get_EAR_ratio(eye):
    # Calculating EAR ratio from both eyes.
    
    # Vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # return the Eye Aspect Ratio
    return (A + B) / (2.0 * C)

def encode(image):
    # Making new face encodings for new user.
    
    # Get images
    images = list(map(get_images, image))

    # Get encodings of face in photos
    face_encode = []
    face_found = True
    print('Encoding faces. This might take a while...')
    for index, img in enumerate(tqdm(images)):
        try:
            face_encode.append(fr.face_encodings(img, num_jitters=50)[0])
        except Exception as e:
            print("\nException: " + e)
            face_found = False
            break

    if face_found:
        print('Saving face encodings, please wait...')
        # Save Face encoding
        np.save('assets/face_encodings.npy', face_encode)
        print(f"Data saved for {index+1} images...")

        # Saving pickle file for number of files
        with open('assets/n_people.pk', 'wb') as pickle_file:
            pickle.dump(len(image), pickle_file)
            
        print('Face encoding complete.')