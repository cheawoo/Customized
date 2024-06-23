import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

# Load the pre-trained model and tokenizer
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Define the image URL
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'

# Download the image content
image_data = requests.get(img_url, stream=True).content

raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
text = "a photography of"  # Conditional captioning text

# Decode the image data and convert to NumPy array
image_array = np.frombuffer(image_data, np.uint8)

# Load the image using OpenCV
img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# Measure inference time for conditional image captioning
start_time = time.time()
inputs = processor(raw_image, text, return_tensors="pt")
inputs = {k: v.to("cpu") for k, v in inputs.items()}  # Ensure tensors are on CPU
out = model.generate(**inputs)
caption_with_text = processor.decode(out[0], skip_special_tokens=True)
conditional_time = time.time() - start_time

# Measure inference time for unconditional image captioning
start_time = time.time()
inputs = processor(raw_image, return_tensors="pt")
inputs = {k: v.to("cpu") for k, v in inputs.items()}  # Ensure tensors are on CPU
out = model.generate(**inputs)
caption_without_text = processor.decode(out[0], skip_special_tokens=True)
unconditional_time = time.time() - start_time

# Display the image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying with matplotlib
plt.axis('off')  # Hide the x and y axis labels

# Print captions with timing information
print("Conditional Caption:", caption_with_text)
print("Unconditional Caption:", caption_without_text)
print(f"Conditional Inference Time: {conditional_time:.4f} seconds")
print(f"Unconditional Inference Time: {unconditional_time:.4f} seconds")

# Display the plot
plt.show()