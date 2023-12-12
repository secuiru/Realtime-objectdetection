import cv2
import cvzone

def face_overlay_app():
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    overlay = cv2.imread('pekka.png', cv2.IMREAD_UNCHANGED)

    while True:
        _, frame = cap.read()
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray_scale)
        
        for (x, y, w, h) in faces:
            overlay_resize = cv2.resize(overlay, (int(w*1.3), int(h*1.6)))
            frame = cvzone.overlayPNG(frame, overlay_resize, [x-17, y-55])

        cv2.imshow('', frame)
        
        if cv2.waitKey(10) == ord('q'):
            break

    # Vapauta resurssit
    cap.release()
    cv2.destroyAllWindows()

# Käynnistä applikaatio
face_overlay_app()