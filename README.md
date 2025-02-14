---
language:
- en
license: apache-2.0
datasets:
- allenai/olmOCR-mix-0225
base_model:
- Qwen/Qwen2-VL-7B-Instruct
library_name: transformers
---

<img alt="olmOCR Logo" src="https://huggingface.co/datasets/allenai/blog-images/resolve/main/olmocr/olmocr.png" width="242px" style="margin-left:'auto' margin-right:'auto' display:'block'">

# olmOCR-7B-0225-preview

This is a preview release of the olmOCR model that's fine tuned from Qwen2-VL-7B-Instruct using the 
[olmOCR-mix-0225](https://huggingface.co/datasets/allenai/olmOCR-mix-0225) dataset.

Quick links:
- 📃 [Paper](link-to-paper)
- 🤗 [Dataset](https://huggingface.co/allenai/olmOCR-mix-0225)
- 🛠️ [Code](https://github.com/allenai/olmocr)
- 🎮 [Demo](https://olmocr.allenai.org/)

The best way to use this model is via the [olmOCR toolkit](https://github.com/allenai/olmocr).

## Usage

This model expects as input a single document image, rendered such that the longest dimension is 1024 pixels.

The prompt must then contain the additional metadata from the document, and the easiest way to generate this


## Manual Prompting

If you want to prompt this model manually, please see the code below.

In normal usage, the olmOCR toolkit builds the prompt by rendering the PDF page, and
extracting relevant text blocks and image metadata. To duplicate that you will need to

```bash
pip install olmocr
```

and then run the following sample code.


```python
import torch
import base64
import json
import urllib.request

from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

# Initialize the model
model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Grab a sample PDF
urllib.request.urlretrieve("https://molmo.allenai.org/paper.pdf", "./paper.pdf")

# Render page 1 to an image
image_base64 = render_pdf_to_base64png("./paper.pdf", 1, target_longest_image_dim=1024)

# Build the prompt, using document metadata
anchor_text = get_anchor_text("./paper.pdf", 1, pdf_engine="pdfreport", target_length=4000)
prompt = build_finetuning_prompt(anchor_text)

# Build the full prompt
messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]

# Apply the chat template and processor
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

inputs = processor(
    text=[text],
    images=[main_image],
    padding=True,
    return_tensors="pt",
)
inputs = {key: value.to(device) for (key, value) in inputs.items()}


# Generate the output
output = model.generate(
            **inputs,
            temperature=0.8,
            max_new_tokens=50,
            num_return_sequences=1,
            do_sample=True,
        )

# Decode the output
prompt_length = inputs["input_ids"].shape[1]
new_tokens = output[:, prompt_length:]
text_output = processor.tokenizer.batch_decode(
    new_tokens, skip_special_tokens=True
)

print(text_output)
```

## License and use

olmOCR is licensed under the Apache 2.0 license.
olmOCR is intended for research and educational use.
For more information, please see our [Responsible Use Guidelines](https://allenai.org/responsible-use).
