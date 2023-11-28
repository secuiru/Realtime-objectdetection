import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image as Img
from PIL import ImageTk
import numpy as np
import torch
import tkinter as tk
from tkinter import *
import time
import easyocr
from translate import Translator


to_lang = 'en'
translator= Translator(to_lang="en", from_lang='autodetect')

ikkuna = tk.Tk()
ikkuna.title("Object detect")
ikkuna.geometry("1400x512")
app = Frame(ikkuna, bg="black")
app.grid()

lmain = Label(app)
lmain.grid()

text_box=Text(ikkuna,height=3,width=15)
text_scroll=Scrollbar(ikkuna, orient='vertical',command=text_box.yview)
text_scroll.grid(row=0,column=2,sticky=tk.NS)
text_box=Text(yscrollcommand=text_scroll.set)


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
translate_var=False

reader = easyocr.Reader(['en'])


def video_stream():
    global fps, video_on

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
    if video_on:
        ikkuna.after(10, video_stream)

def text_detection():
    global fps, video_on,translate_var
    start_time = time.time()
    ret, frame = cap.read()

    results = reader.readtext(frame)

    for (bbox, text, prob) in results:
        if prob >0.8:#THRESHOLD
            #print("[INFO] {:.4f}: {}".format(prob, text))
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))
            cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
            cv2.putText(frame, text, (tl[0], tl[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if translate_var:
                text_box.insert(tk.END,text+"\n")
            else:
                translation = translator.translate(text)
                #print(translation)
                text_box.insert(tk.END,translation+"\n")


    if not video_on:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Img.fromarray(img)
        im_tk = ImageTk.PhotoImage(im_pil)
    	
        lmain.img = im_tk
        lmain.configure(image=im_tk)
        fps_r = round(fps, 1)
        fpsdisplay = tk.Label(ikkuna, text=f"FPS: {fps_r}")
        fpsdisplay.place(x=550, y=0)
        ikkuna.after(10, text_detection)
        end_time = time.time()
        fps = 1 / np.round(end_time - start_time, 3)
        fps_r = round(fps, 1)
        fpsdisplay = tk.Label(ikkuna, text=f"FPS: {fps_r}")
        fpsdisplay.place(x=550, y=0)




def toggle_mode():
    global video_on
    video_on = not video_on
    if video_on:
        current_mode_label.config(text="Current Mode: Video")
        video_stream()
    else:
        current_mode_label.config(text="Current Mode: Text")
        text_detection()

def translate_mode():
    global translate_var
    if translate_var:
        translate_var = False
    else:
            translate_var=True

video_button = Button(ikkuna, text="Toggle mode", command=toggle_mode)
video_button.place(y=20, x=650)

translate_button = Button(ikkuna, text="Translate", command=translate_mode)
translate_button.place(y=20, x=730)

text_box.grid(row=0,column=1)
text_box.insert(tk.END,"")

video_stream()

ikkuna.mainloop()