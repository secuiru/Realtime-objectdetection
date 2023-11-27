import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image, ImageTk
import numpy as np
import torch
import tkinter as tk
from tkinter import *
import time
import easyocr

ikkuna = tk.Tk()
ikkuna.title("Object detect")
ikkuna.geometry("1000x512")
app = Frame(ikkuna, bg="black")
app.grid()

lmain = Label(app)
lmain.grid()

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.85)
model.to(device)
model.eval()

fps = 0
cap = cv2.VideoCapture(0)
cap.set(3, 512)
cap.set(4, 512)

video_on = True  
current_mode_label = tk.Label(ikkuna, text="Current Mode: Video Stream")
current_mode_label.place(x=700, y=0)


captured_image = None

reader = easyocr.Reader(['en'])


def video_stream():
    global fps
    ret, img = cap.read()
    start_time = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = to_tensor(img).to(device) * 255
    img = img.type(torch.uint8)

    preprocess = weights.transforms()

    batch = [preprocess(img)]
    with torch.no_grad():
        prediction = model(batch)[0]

    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                               labels=labels,
                               colors="red",
                               width=4, font='calibri.ttf', font_size=30)
    im = to_pil_image(box.detach().cpu())

    im_tk = ImageTk.PhotoImage(im)

    lmain.img = im_tk
    lmain.configure(image=im_tk)
    end_time = time.time()
    fps = 1 / np.round(end_time - start_time, 3)
    fps_r = round(fps, 1)
    fpsdisplay = tk.Label(ikkuna, text=f"FPS: {fps_r}")
    fpsdisplay.place(x=550, y=0)
    ikkuna.update_idletasks()
    if video_on:
        ikkuna.after(10, video_stream)

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def text_detection():
    
    
    global fps
    ret, img = cap.read()
    start_time = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    



    results = reader.readtext(img)

    for (bbox, text, prob) in results:
      
        print("[INFO] {:.4f}: {}".format(prob, text))
       
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))
 
        text = cleanup_text(text)

        cv2.rectangle(img, tl, br, (0, 255, 0), 2)
        cv2.putText(img, text, (tl[0], tl[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

   

    

def toggle_mode():
    global video_on
    video_on = not video_on
    if video_on:
        current_mode_label.config(text="Current Mode: Video")
        video_stream()
    else:
        current_mode_label.config(text="Current Mode: Text")
        text_detection()

video_button = Button(ikkuna, text="Toggle Mode", command=toggle_mode)
video_button.place(y=0, x=200)

video_stream()

ikkuna.mainloop()