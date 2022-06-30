# Imports
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import datetime as dt
from PIL import Image, ImageTk
import cv2
import os
from glob import glob
from pathlib import Path
import face_recognition as fr
import definitions as d

img_dir = "photo"

def init():
    # Window generation
    main = Tk()
    main.title("Face Identity System - UI") # Window title
    main.resizable(False, False)
    #main.geometry("640x480")

    # Definitions 
    ## Attendance Mode
    #def attendance():
    #    if(os.path.exists('assets/n_people.pk') == False):
    #        messagebox.showerror("No User", "You haven't make a user yet!")
    #    else:
    #        os.system("python main.py")
    #        main.destroy()
        
    ## New User
    def new_user():
        newUser = Toplevel(main)
        newUser.title("New User")
        newUser.resizable(False, False)
        #newUser.geometry("400x400")
        
        # Naming
        Label(newUser, text="Enter the full name of new user").pack()
        Label(newUser, text="(ex: John Helena Carter)").pack()
        
        temp = Entry(newUser)
        temp.pack()
        
        Button(newUser, text="Next", command=lambda: new_user_camera(temp.get())).pack()
        Button(newUser, text="Close", command=newUser.destroy).pack()
        
    def new_user_camera(name):
        # Debug
        #print(name)
        
        if(name == ""):
            messagebox.showerror("Missing Value", "You have not filled the name box!")
        else:
            # Photo taking
            cam = cv2.VideoCapture(0)
            while(True):
                # Capture camera frame by frame
                ret, frame = cam.read()
                
                # Get face locations, landmarks and encodings
                frame_face_loc = fr.face_locations(frame)
                frame_face_landmarks = fr.face_landmarks(frame, frame_face_loc)
                frame_face_encode = fr.face_encodings(frame, frame_face_loc)
                
                for index, (loc, encode, landmark) in enumerate(zip(frame_face_loc, frame_face_encode, frame_face_landmarks)):
                    
                    # Show display output
                    #cv2.rectangle(frame,(loc[3],loc[0]),(loc[1],loc[2]), (0,0,0),2) # top-right, bottom-left
                    cv2.imshow('Camera Output', frame)
                    
                    # User creation
                    if cv2.waitKey(100) & 0xFF == ord(' '):
                        cv2.imwrite("photo\ " + name + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
                        print("Photo saved")
                        cv2.destroyWindow('Camera Output')
                        cam.release()
                        
                    # Press 'q' is quit
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        cv2.destroyWindow('Camera Output')
                        cam.release()
                
                    
       
    ## User List
    def list_user():
        listUser = Toplevel(main)
        listUser.title("User List")
        Label(listUser, text="List of Registered Users").pack()
        
        listbox = Listbox(listUser, width=50)
        listbox.pack()
        
        # List folder
        img_list = glob(img_dir + '/*.*')
        names = list(map(d.get_names, img_list))
        
        # List registered users
        n = -1
        for i in names:
            n = n+1
            listbox.insert(n, i)
            
        # View photo button
        Button(listUser, text="View Photo")
        
        # Close button
        Button(listUser, text="Close", command=listUser.destroy).pack()
        
    def convertTuple(tup):
        # initialize an empty string
        str = ''
        for item in tup:
            str = str + item
        return str
    
    def see_photo(name):
        print(name)
    
    def list_attendance():
        listAttend = Toplevel(main)
        listAttend.title("Attendance List")
        
        listbox = Listbox(listAttend, width=50)
        listbox.pack()
        
        # Read CSV file
        
        # Close button
        Button(listAttend, text="Close", command=listAttend.destroy).pack()
    
    ## Exit program
    def quit():
         print("Quitting... Good night!")
         exit(0)

    # Elements
    date = dt.datetime.now()

    Label(
        main, 
        text=f"Today's date is: {date:%A, %B %d, %Y}", font="Calibri, 15"
    ).pack()

    #label_time = Label(
    #    main,
    #    text=f"{date: %H:%M:%S %p}", font="Calibri, 20"
    #).pack()

    #attendance_button = Button(
    #    text="Attendance Mode",
    #    command = attendance
    #)
    #.pack()
    Label(main, text="What do you want to do?").pack()
    

    user_new_button = Button(
        text="New User",
        command = new_user
    ).pack()

    user_list_button = Button(
        text="User List",
        command = list_user
    ).pack()
    
    attendance_list_button = Button(
        text="Attendance List",
        command = list_attendance
    )

    quit_button = Button(
        text="Quit",
        command = quit
    ).pack()


    # Open window
    main.mainloop()
    

# Debug purposes
#init()