import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Initialize the emotion model
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

# Emotion dictionary with correct keys
emotion_dict = {
    0: "   Angry   ",
    1: "Disgusted",
    2: "  Fearful  ",
    3: "   Happy   ",
    4: "  Neutral  ",
    5: "    Sad    ",
    6: "Surprised"
}

# Emoji images mapped correctly to unique keys
emoji_dist = {
    0: "./emojis/angry.png",
    1: "./emojis/disgusted.png",
    2: "./emojis/fearful.png",
    3: "./emojis/happy.png",
    4: "./emojis/neutral.png",
    5: "./emojis/sad.png",
    6: "./emojis/surprised.png"
}

# Global variables for frames and emotion state
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]

# Function to show the video from webcam
def show_vid():
    global cap1, last_frame1, show_text
    cap1 = cv2.VideoCapture(0)  # Start the webcam
    if not cap1.isOpened():
        print("Can't open the camera")
        return

    # Face detection setup
    bounding_box = cv2.CascadeClassifier('/home/shivam/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    while True:
        flag1, frame1 = cap1.read()
        frame1 = cv2.resize(frame1, (600, 500))  # Resize frame
        gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # Process each detected face
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)

            # Get the predicted emotion
            maxindex = int(np.argmax(prediction))
            show_text[0] = maxindex  # Update the emotion index

        # Display video
        pic = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        # Update after every 10ms
        lmain.after(10, show_vid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit on 'q'

    cap1.release()

# Function to show corresponding emoji
def show_vid2():
    global show_text, emoji_dist

    # Load the emoji image based on emotion prediction
    frame2 = cv2.imread(emoji_dist[show_text[0]])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(frame2)
    imgtk2 = ImageTk.PhotoImage(image=img2)

    # Update the emoji image
    lmain2.imgtk2 = imgtk2
    lmain2.configure(image=imgtk2)

    # Update the emotion label
    lmain3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))

    # Update after every 10ms
    lmain2.after(10, show_vid2)

# Main Tkinter setup
if _name_ == '_main_':
    root = tk.Tk()
    img = ImageTk.PhotoImage(Image.open("logo.png"))
    heading = Label(root, image=img, bg='black')
    heading.pack()

    heading2 = Label(root, text="Photo to Emoji", pady=20, font=('arial', 45, 'bold'), bg='black', fg='#CDCDCD')
    heading2.pack()

    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')

    lmain.pack(side=LEFT)
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900, y=350)

    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'

    exitbutton = Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)

    # Start video display
    show_vid()
    show_vid2()

    root.mainloop()