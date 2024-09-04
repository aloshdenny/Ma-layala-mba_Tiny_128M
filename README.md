---
license: mit
datasets:
- VishnuPJ/Malayalam_CultureX_IndicCorp_SMC
library_name: transformers
language:
- ml
tags:
- mamba
- ssm
- s6
- jamba
- llm
- state space models
- malayalam
- indic
---

# Ma-layala-mba

Welcome to Ma-layala-mba, a base Indic language model designed to push the boundaries of NLP for Indian languages. It is based on the Mamba series of state space models.

![Thumbnail](thumbnail.jpg)

## Model Description

Ma-layala-mba is a state-of-the-art S6 SSM model specifically crafted for the South Indian regional and state language of Kerala: Malayalam. It integrates traditional Attention mechanisms with innovative approaches such as MLPs and State Space Models (SSMs) to handle complex linguistic features and achieve high accuracy in language understanding and generation.

- **Model Type**: A 128M Jamba model finetuned on ~1 million samples of Malayalam prompt-response pairs from a subset of the IndicCorp Dataset
- **Language(s)**: Malayalam
- **License**: GNU General Public License v3.0
- **Training Precision**: bfloat16

## Example Usage

Here's a quick example to get you started with the Ma-layala-mba model:

```python
from transformers import MaLayalaMbaForCausalLM, AutoTokenizer, pipeline

model = MaLayalaMbaForCausalLM.from_pretrained(
    "aoxo/Ma-layala-mba_Tiny_128M",
    # load_in_8bit=True, # Set this depending on the GPU you have
    torch_dtype=torch.bfloat16,
    device_map={"": 0}, # Set this depending on the number of GPUs you have
    local_files_only=False # Optional
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("aoxo/Ma-layala-mba_Tiny_128M")

input_ids = tokenizer("മലയാളം പര്യായപദങ്ങളിൽ ഒരു പരീക്ഷ പേപ്പർ ഉണ്ടാക്കുക", return_tensors='pt').to(model.device)["input_ids"]

outputs = model.generate(input_ids, max_new_tokens=100)

print(tokenizer.batch_decode(outputs))
```

### Example Output:

```
മലയാളം പര്യായപദങ്ങളിൽ ഒരു പരീക്ഷ പേപ്പർ ഉണ്ടാക്കുക

a. വലിയ - __________
b. രസം - __________
c. സുഖം - __________
d. പ്രകാശം - __________
e. വേഗം - __________
```

## Usage Note

Please be aware that this model has not undergone comprehensive detoxification or censorship. While it exhibits strong linguistic capabilities, there is a possibility of generating content that may be deemed harmful or offensive. We advise users to apply discretion and closely monitor the model's outputs, especially in public or sensitive settings.

## Meet the Developers

- **[Alosh Denny](https://x.com/AloshDenny)**