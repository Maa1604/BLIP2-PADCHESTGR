from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("fatehmujtaba/blip2-opt-2.7b-for-Chest-Xray")
model = Blip2ForConditionalGeneration.from_pretrained(
    "fatehmujtaba/blip2-opt-2.7b-for-Chest-Xray", load_in_8bit=True, device_map={"": 0}, dtype=torch.float16
)  # doctest: +IGNORE_RESULT

url = "thus.jpg"
image = Image.open(url)

inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)