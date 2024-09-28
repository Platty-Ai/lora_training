# lora training for SDXL

## Running locally with PyTorch

### Installing the dependencies

Before executing the scripts, ensure that you have installed the necessary training dependencies for the library:

**Important Note**

To ensure compatibility with the latest versions of the example scripts, we strongly recommend **installing the library from source** and keeping your installation up-to-date, as we frequently update the example scripts and include specific requirements. Follow these steps in a new virtual environment:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then install the pip packages
```bash
pip install -r requirements.txt
```

Alternatively, for a default Accelerate configuration without having to answer environment-related questions:

```bash
accelerate config default
```

When running `accelerate config`, enabling torch compile mode by setting it to True can result in significant speed improvements. Additionally, for LoRA training, we use the PEFT library as the backend, so ensure that `peft>=0.6.0` is installed in your environment.

## To train the model
- MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
- INSTANCE_DIR = "YOUR_FOLDER_NAME" (For image dataset)
- OUTPUT_DIR = "OUTPUT_FOLDER" (For the output model directory) 
- VAE_PATH = "madebyollin/sdxl-vae-fp16-fix"

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="zetty"
export OUTPUT_DIR="lora-trained-zetty-xl"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of zetty" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of zetty playing poker" \
  --validation_epochs=25 \
  --seed="0" \
```

### To train the model with < 16GB VRAM

```diff
+  --enable_xformers_memory_efficient_attention \
+  --gradient_checkpointing \
+  --use_8bit_adam \
+  --mixed_precision="fp16" \
```

and making sure that you have the following libraries installed:

```
bitsandbytes>=0.40.0
xformers>=0.0.20
```
### Upload the safetensors file to Novita AI
Navigate to your "OUTPUT_DIR" folder and locate the safetensors file. Upload this file to the Novita console by visiting [Novita AI Model API Console](https://novita.ai/model-api/console/model).

### To test the Inference locally

Once training is done, we can perform inference like so:

```python
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline
import torch

lora_model_id = <"lora-trained-zetty-xl">
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_model_id)
image = pipe("A picture of zetty playing poker", num_inference_steps=25).images[0]
image.save("zetty.png")
```

It can further refine the outputs with the Refiner in SDXL

```python
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch

lora_model_id = <"lora-trained-zetty-xl">
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

# Load the base pipeline and load the LoRA parameters into it.
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_model_id)

# Load the refiner.
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
refiner.to("cuda")

prompt = "A picture of zetty playing poker"
generator = torch.Generator("cuda").manual_seed(0)

# Run inference.
image = pipe(prompt=prompt, output_type="latent", generator=generator).images[0]
image = refiner(prompt=prompt, image=image[None, :], generator=generator).images[0]
image.save("refined_zetty.png")
```