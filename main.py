import os
print(os.path.exists("C:/Users/Administrator/.deepface/weights/facenet_weights.h5"))


import threading  
import cv2  
from deepface import DeepFace 


cv2.setUseOptimized(True)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30) 

Counter = 0
face_match = False
reference_img = cv2.imread("refrence.jpg")

def check_face(frame):
    global face_match
    try:
        result = DeepFace.verify(frame, reference_img.copy(),model_name="Facenet")
        face_match = result["verified"]
    except Exception as e:
        print(f"Erreur DeepFace: {e}")
        face_match = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire la vidéo.")
        break

    
    if Counter % 100 == 0:
        try:
            threading.Thread(target=check_face, args=(frame.copy(),), daemon=True).start()
        except Exception as e:
            print(f"Erreur threading : {e}")

    Counter += 1
    if face_match:
        cv2.putText(frame, "MATCH!", (20, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0,255,0),3)
    else:
        cv2.putText(frame, "NO MATCH!", (20, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0,0,255),3)
    

    cv2.imshow("video", frame)


    key=cv2.waitKey(1)
    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()