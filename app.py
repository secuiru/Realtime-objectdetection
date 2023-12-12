import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image as Img
from PIL import ImageTk
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk
from tkinter import * 
import time
import easyocr
from translate import Translator
import cvzone


to_lang = ''
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

filtterilista = ["Will Smith", "Aurinkolasit", "Pekka", "Trump"]

lista = ['en','se','fi','zh','de']
clicked = StringVar() 
clicked.set('en')

to_lang = clicked.get()
translator= Translator(to_lang= to_lang, from_lang='autodetect')

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.85)
model.to(device)
model.eval()

def update_language(*args):
    global to_lang, translator
    to_lang = clicked.get()
    translator = Translator(to_lang=to_lang, from_lang='autodetect')


fps = 0
cap = cv2.VideoCapture(0)
cap.set(3, 512)
cap.set(4, 512)

boolblur = False
video_on = True  
current_mode_label = tk.Label(ikkuna, text="Current Mode: Video Stream")
current_mode_label.place(x=700, y=0)


captured_image = None
translate_var=False

reader = easyocr.Reader(['en'])


        
    
def video_stream():
    global fps, video_on, cap, boolblur

    ret, img = cap.read()
    start_time = time.time()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = to_tensor(img_rgb).to(device) * 255
    img_tensor = img_tensor.type(torch.uint8)

    preprocess = weights.transforms()
    batch = [preprocess(img_tensor)]
    
    with torch.no_grad():
        prediction = model(batch)[0]

    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    
    mask = np.zeros_like(img_rgb, dtype=np.uint8)
    
    for box in prediction["boxes"]:
        x, y, w, h = map(int, box)
        mask[y:y+h, x:x+w] = 255

    mask_inv = cv2.bitwise_not(mask)

    if boolblur:
        blurred_img = cv2.GaussianBlur(img_rgb, (25, 25), 0)  
        img_rgb[mask_inv > 0] = blurred_img[mask_inv > 0]

  
    for box, label in zip(prediction["boxes"], labels):
        x, y, w, h = map(int, box)
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 4)
        cv2.putText(img_rgb, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    pil_img = to_pil_image(img_rgb)
    im_tk = ImageTk.PhotoImage(pil_img)

    lmain.img = im_tk
    lmain.configure(image=im_tk)
    
    end_time = time.time()
    fps = 1 / np.round(end_time - start_time, 3)
    fps_r = round(fps, 1)
    fpsdisplay.config(text=f"FPS: {fps_r}")

    if video_on:
        ikkuna.after(10, video_stream)


def filtterit():
    global video_on, cap
    video_on = not video_on
    overlay = cv2.imread('pekka.png', cv2.IMREAD_UNCHANGED)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    _, frame = cap.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    faces = cascade.detectMultiScale(frame, 1.1, 8)
    for (x, y, w, h) in faces:
        overlay_resize = cv2.resize(overlay, (int(w*1.3), int(h*1.6)))
        frame = cvzone.overlayPNG(frame, overlay_resize, [x-17, y-55])  
        

    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)

    pil_img = to_pil_image(frame)
    im_tk = ImageTk.PhotoImage(pil_img)
    
    lmain.img = im_tk
    lmain.configure(image=im_tk)
    
    ikkuna.after(10, filtterit)


def text_detection():
    global fps, video_on,translate_var,cap
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
                text_box.insert(tk.END,translation+"\n")

    text_box.yview(tk.END)

    if not video_on:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Img.fromarray(img)
        im_tk = ImageTk.PhotoImage(im_pil)
    	
        lmain.img = im_tk
        lmain.configure(image=im_tk)
        ikkuna.after(10, text_detection)
        end_time = time.time()
        fps = 1 / np.round(end_time - start_time, 3)
        fps_r = round(fps, 1)
        fpsdisplay.config(text=f"FPS: {fps_r}")




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
        translate_button.config(text="Translating on")
    else:
        translate_var=True
        translate_button.config(text="Translating off")

def input_changed(event):
    global cap
    selection = input_device_list.get()
    selection =int(selection)
    cap = cv2.VideoCapture(selection)
    cap.set(3, 512)
    cap.set(4, 512)
   
def effects():
    global boolblur
    if boolblur:
        boolblur = False
    else:
        boolblur = True


stringvariable=tk.StringVar()
input_device_list = ttk.Combobox(ikkuna,values=["0", "1", "2", "3"],textvariable=stringvariable)

video_button = Button(ikkuna, text="Toggle mode", command=toggle_mode)
video_button.place(y=20, x=650)

blur_button = Button(ikkuna, text="Blur", command=effects)
blur_button.place(y=20, x=870)

translate_button = Button(ikkuna, text="Translating on", command=translate_mode)
translate_button.place(y=20, x=730)

text_box.grid(row=0,column=1)
text_box.insert(tk.END,"")

clicked = StringVar() 
clicked.set('en')
clicked.trace_add('write', update_language)
option_menu = OptionMenu(ikkuna, clicked, *lista)
option_menu.place(x=815, y=17)


klikattu = StringVar() 


#valinta_menu = OptionMenu(ikkuna, klikattu, *filtterilista)
#valinta_menu.place(x=905, y=20)
valinta_button = Button(ikkuna, text="filtteri", command=filtterit)
valinta_button.place(y=20, x=930)

fpsdisplay = tk.Label(ikkuna, text=f"FPS:--")
fpsdisplay.place(x=600, y=0)

input_device = tk.Label(ikkuna, text="Input device")
input_device.place(x=1050, y=0)
input_device_list.place(x=1050, y=24)
input_device_list.bind("<<ComboboxSelected>>", input_changed)
input_device_list.current()

video_stream()

ikkuna.mainloop()
