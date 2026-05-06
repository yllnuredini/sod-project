import time
import torch
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms
from sod_model import SODModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SODModel().to(device)
checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image):
    original_image = image.convert("RGB")
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    start_time = time.time()

    with torch.no_grad():
        output = model(input_tensor)

    inference_time = (time.time() - start_time) * 1000

    pred_mask = output.squeeze().cpu().numpy()
    pred_mask = (pred_mask * 255).astype(np.uint8)

    mask_image = Image.fromarray(pred_mask).resize(original_image.size)

    image_np = np.array(original_image).astype(np.float32)
    mask_np = np.array(mask_image).astype(np.float32) / 255.0

    overlay = image_np.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask_np * 120, 0, 255)
    overlay = overlay.astype(np.uint8)

    overlay_image = Image.fromarray(overlay)

    return mask_image, overlay_image, f"Inference time: {inference_time:.2f} ms"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Input Image"),
    outputs=[
        gr.Image(type="pil", label="Predicted Saliency Mask"),
        gr.Image(type="pil", label="Overlay Visualization"),
        gr.Textbox(label="Inference Time")
    ],
    title="Salient Object Detection Demo",
    description="Upload an image to generate the saliency mask and overlay visualization."
)

demo.launch(share=True)
