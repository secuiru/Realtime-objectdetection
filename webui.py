import gradio as gr
import torch
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

#print(torch.cuda.is_available())

def object_detect(input_img):
    img = read_image(input_img).to(device)  # Move input image tensor to the same device as the model

    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8).to(device)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img)]

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4, font=None, font_size=30)
    im = to_pil_image(box.detach())

    return im



with gr.Blocks() as demo:
    gr.Interface(object_detect, gr.Image(type="filepath"), "image")
    gr.Interface(object_detect, gr.Image(type="filepath"), "image")

print(f"Using {device}")

demo.launch()
