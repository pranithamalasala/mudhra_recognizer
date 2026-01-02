from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import mediapipe
from math import hypot
import cvzone
import numpy as np
from email.message import EmailMessage
import smtplib

app = Flask(__name__)

# --- CONFIGURATION ---
cap = cv2.VideoCapture(0)
initHand = mediapipe.solutions.hands
mainHand = initHand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
draw = mediapipe.solutions.drawing_utils

# Global variables for logic
email_list = []

def fingers(landmarks):
    fingerTips = []  
    tipIds = [4, 8, 12, 16, 20] 
    
    # Thumb
    if landmarks[tipIds[0]][1] > landmarks[tipIds[0] - 1][1]:
        fingerTips.append(1)
    else:
        fingerTips.append(0)
    
    # Other 4 fingers
    for id in range(1, 5):
        if landmarks[tipIds[id]][2] < landmarks[tipIds[id] - 3][2]: 
            fingerTips.append(1)
        else:
            fingerTips.append(0)

    return fingerTips

def handLandmarks(colorImg):
    landmarkList = []  
    landmarkPositions = mainHand.process(colorImg)  
    landmarkCheck = landmarkPositions.multi_hand_landmarks  
    if landmarkCheck: 
        for hand in landmarkCheck:  
            for index, landmark in enumerate(hand.landmark): 
                draw.draw_landmarks(colorImg, hand, initHand.HAND_CONNECTIONS)  
                h, w, c = colorImg.shape  
                centerX, centerY = int(landmark.x * w), int(landmark.y * h)  
                landmarkList.append([index, centerX, centerY])  
    return landmarkList

def generate_frames():
    global email_list
    while True:
        success, img = cap.read()
        if not success:
            break
            
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lmList = handLandmarks(imgRGB)

        if len(lmList) != 0:
            x1, y1 = lmList[4][1:]  # Thumb
            x2, y2 = lmList[8][1:]  # Index
            x3, y3 = lmList[12][1:]  # Middle
            x4, y4 = lmList[16][1:]  # Ring
            x5, y5 = lmList[20][1:]  # Pinky
            finger = fingers(lmList)

            # Draw Skeleton
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 3)
            cv2.line(img, (x3, y3), (x4, y4), (255, 0, 255), 3)
            cv2.line(img, (x4, y4), (x5, y5), (255, 255, 255), 3)

            # Calculate Distances
            lengththind = hypot(x2 - x1, y2 - y1)
            lengthindmid = hypot(x3 - x2, y3 - y2)
            lengthmidrin = hypot(x4 - x3, y4 - y3)
            lengthrinthu = hypot(x4 - x1, y4 - y1)
            lengthmidthu = hypot(x3 - x1, y3 - y1)

            # --- MUDRA LOGIC ---
            # 1. Pataka
            if finger[1] == 1 and finger[0] == 1 and finger[2]==1 and finger[3] == 1 and finger[4]==1 and lengththind<150:  
                cvzone.putTextRect(img, "Pataka", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
                email_list.append("Pataka")
            # 2. Tripataka
            elif finger[1] == 1 and finger[2]==1 and finger[3] == 0 and finger[4]==1 and lengthrinthu>40:
                cvzone.putTextRect(img, "Tripataka", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
                email_list.append("Tripataka")
            # 3. Shikaram
            elif finger[0] == 1 and finger[1] == 0 and finger[2]==0 and finger[3] == 0 and finger[4]==0 :  
                cvzone.putTextRect(img, "Shikaram", (50, 150), scale=4, thickness=5, colorR=(0, 200, 0), offset=20) 
                email_list.append("Shikaram")
            # 4. Ardhapataka
            elif finger[0] == 0 and finger[1] == 1 and finger[2]==1 and finger[3] == 0 and finger[4]==0 :  
                cvzone.putTextRect(img, "Ardhapataka", (50, 150), scale=4, thickness=5, colorR=(0, 200, 0), offset=20)
                email_list.append("Ardhapataka")
            # 5. Kartharimukha
            elif finger[0] == 0 and finger[1] == 1 and finger[2]==1 and finger[3] == 0 and finger[4]==0 and 19<=lengthindmid<=94 :  
                cvzone.putTextRect(img, "Kartharimukha", (50, 150), scale=4, thickness=5, colorR=(0, 200, 0), offset=20) 
                email_list.append("Kartharimukha")
            # 6. Mayura
            elif finger[1] == 1 and finger[2]==1 and finger[3] == 0 and finger[4]== 1 and 12<=lengthrinthu<=40 :  
                cvzone.putTextRect(img, "Mayura", (50, 150), scale=4, thickness=5, colorR=(0, 200, 0), offset=20)
                email_list.append("Mayura")
            # 7. Ardhachandra
            elif finger[1] == 1 and finger[0] == 1 and finger[2]==1 and finger[3] == 1 and finger[4]==1 and 150<=lengththind<=300:  
                cvzone.putTextRect(img, "Ardhachandra", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
                email_list.append("Ardhachandra")
            # 8. Arala
            elif finger[1] == 0 and finger[0] == 1 and finger[2]==1 and finger[3] == 1 and finger[4]==1 :  
                cvzone.putTextRect(img, "Arala", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
                email_list.append("Arala")
            # 9. Katakamukaha
            elif finger[3] == 1 and finger[4]==1 and 7<=lengthmidthu<= 40 and 5<=lengththind<= 30 and 10<=lengthindmid<=33:  
                cvzone.putTextRect(img, "Katakamukaha", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)    
                email_list.append("Katamukaha")
            # 10. Simhamukaha
            elif finger[1] == 1 and finger[4]==1 and 3<=lengthrinthu<= 30 and 1<=lengthmidthu<=15 and 1<=lengthmidrin<=25:  
                cvzone.putTextRect(img, "Simhamukaha", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)   
                email_list.append("Simhamukaha") 
            # 11. Kapitha
            elif finger[2] == 0 and finger[3]==0 and finger[4] ==0 and finger[0] == 1 and 10<=lengththind<=40:  
                cvzone.putTextRect(img, "Kapitha", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)    
                email_list.append("Kapitha")
            # 12. Mushti
            elif finger[1] == 0 and finger[0] == 0 and finger[2]==0 and finger[3] == 0 and finger[4]==0 and 3<=lengththind<=15:  
                cvzone.putTextRect(img, "Mushti", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
                email_list.append("Mushti")
            # 13. Soochi
            elif finger[1] == 1 and finger[0] == 0 and finger[2]==0 and finger[3] == 0 and finger[4]==0:  
                cvzone.putTextRect(img, "Soochi", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
                email_list.append("Soochi")
            # 14. Chandrakala
            elif finger[1] == 1 and finger[0] == 1 and finger[2]==0 and finger[3] == 0 and finger[4]==0:  
                cvzone.putTextRect(img, "Chandrakala", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
                email_list.append("Chandrakala")
            # 15. Mrigashirsha
            elif finger[1] == 0 and finger[0] == 1 and finger[2]==0 and finger[3] == 0 and finger[4]==1:  
                cvzone.putTextRect(img, "Mrigashirsha", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
                email_list.append("Mrigashirsha")
            # 16. Alapadmakam
            elif finger[1] == 1 and finger[0] == 1 and finger[2]==1 and finger[3] == 1 and finger[4]==0 and 30<=lengththind<=155 and 20<=lengthindmid<=70 and 10<=lengthmidrin<=120:  
                cvzone.putTextRect(img, "Alapadmakam", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
                email_list.append("Alapadmakam")


        # Encode frame for browser
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- FLASK ROUTES ---

@app.route('/')
def home():
    # Renders the Home Page
    return render_template('home.html')

@app.route('/studio')
def studio():
    # Renders the Live Camera Studio
    return render_template('index.html')

@app.route('/library')
def library():
    # Renders the Mudra Library Page
    return render_template('library.html')

@app.route('/about')
def about():
    # Renders the About Page
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/send_email', methods=['POST'])
def send_email():
    global email_list
    recipient_email = request.form.get("recipient_email")
    
    if not recipient_email:
        return redirect(url_for('studio'))

    if email_list:
        final_list = list(set(email_list)) 
        final_result = "\n".join(final_list)

        try:
            msg = EmailMessage()
            msg["Subject"] = "Your Mudra Practice Session Details"
            msg['From'] = "g11ascsannidhay.tws@gmail.com"
            msg["To"] = recipient_email
            msg.set_content(f"Great work practicing today!\n\nThe Mudras you successfully performed are:\n\n{final_result}")

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login("g11ascsannidhay.tws@gmail.com", "pkfxkhqhilllrwyo")
                smtp.send_message(msg)
            
            email_list = []
            
        except Exception as e:
            print(f"Error sending email: {e}")

    return redirect(url_for('studio'))

if __name__ == '__main__':
    app.run(debug=True)