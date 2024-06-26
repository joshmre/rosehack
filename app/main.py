import cv2  

def main():
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 

    camera = cv2.VideoCapture(0) 

    while True:  
        _, image = camera.read()
    
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
        faces = face_classifier.detectMultiScale(gray, 1.3, 5) 
    
        for (x, y, w, h) in faces: 
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)  
    
        cv2.imshow('Computer Vision Hackpack', image) 
    
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): 
            break

    camera.release()
    cv2.destroyAllWindows()