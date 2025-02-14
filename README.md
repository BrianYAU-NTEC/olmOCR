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

```python
image_base64 = [base64 image of PDF rendered down to 1024 px on longest edge]

 "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. Just return the plain text representation of this document as if you were reading it naturally.
Do not hallucinate.
RAW_TEXT_START
Page dimensions: 1836.8x2267.2
[Image 0x0 to 1837x2267]

RAW_TEXT_END"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
```

## License and use

olmOCR is licensed under the Apache 2.0 license.
olmOCR is intended for research and educational use.
For more information, please see our [Responsible Use Guidelines](https://allenai.org/responsible-use).
