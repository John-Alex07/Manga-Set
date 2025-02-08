import time
import os
import streamlit as st
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from PIL import Image

# Set cache directory
cache_dir = "J:\\Important\\VIT Vellore\\SET PROJECT\\cache"

torch.classes.__path__ = []

def load_model(model_name):
    """Load and cache the selected Stable Diffusion model."""
    scheduler = EulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
    model = StableDiffusionPipeline.from_pretrained(model_name, scheduler=scheduler, cache_dir=cache_dir, torch_dtype=torch.float16)
    model.to("cuda")
    return model

@st.cache_resource
def get_models():
    return {
        "Stable Diffusion 2": load_model("stabilityai/stable-diffusion-2"),
        "Stable Diffusion 2.1": load_model("stabilityai/stable-diffusion-2-1")
    }

models = get_models()

# Function to generate an image
def generate_image(prompt, negative_prompt, guidance_scale, num_inference_steps, save_path, model):
    image = model(prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
    image.save(save_path)
    return image

# Streamlit UI
st.title("AI-Powered Scene & Character Generator")

# Sidebar - Gallery of previously generated images
st.sidebar.header("Generated Scenes & Characters")
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

for scene in sorted(os.listdir(output_dir)):
    scene_path = os.path.join(output_dir, scene)
    if os.path.isdir(scene_path):
        st.sidebar.subheader(scene)
        for img_file in sorted(os.listdir(scene_path)):
            img_path = os.path.join(scene_path, img_file)
            st.sidebar.image(img_path, caption=img_file, use_container_width=True)

# Model Choice - Moved to the beginning
model_choice = st.radio("Select Model:", ["Stable Diffusion 2", "Stable Diffusion 2.1"])
model = models[model_choice]

# Scene selection
scene_name = st.text_input("Enter scene name:", "Scene_1")
scene_path = os.path.join(output_dir, scene_name)
os.makedirs(scene_path, exist_ok=True)

# User selection: Characters or Background
mode = st.radio("What would you like to generate?", ["Character", "Background"])

# Common user inputs (for both background and characters)
prompt = st.text_area("Enter your prompt:", "A beautiful futuristic city at sunset")
negative_prompt = st.text_area("Negative prompt (optional):", "blurry, low-quality")
guidance_scale = st.slider("Guidance Scale", 1.0, 15.0, 7.5)
num_inference_steps = st.slider("Inference Steps", 10, 100, 50)

if mode == "Character":
    num_characters = st.number_input("Number of characters to generate:", 1, 10, 1)
    character_images = []

    for i in range(num_characters):
        char_name = st.text_input(f"Character {i+1} Name:", f"Character_{i+1}")
        char_save_path = os.path.join(scene_path, f"{char_name}.png")

        if st.button(f"Generate {char_name}"):
            # Display spinner while generating character image
            with st.spinner(f"Generating {char_name}... Please wait."):
                # Generate the character image using common prompts
                char_image = generate_image(prompt, negative_prompt, guidance_scale, num_inference_steps, char_save_path, model)
            
            # Store and display the generated image
            character_images.append((char_name, char_image))

    # Preview characters
    if character_images:
        st.subheader("Generated Characters")
        for name, img in character_images:
            st.image(img, caption=name, use_container_width=True)

elif mode == "Background":
    background_save_path = os.path.join(scene_path, "Background.png")
    if st.button("Generate Background"):
        # Display spinner while generating background image
        with st.spinner("Generating background... Please wait."):
            # Generate the background image using common prompts
            bg_image = generate_image(prompt, negative_prompt, guidance_scale, num_inference_steps, background_save_path, model)

        # Display the generated background
        st.image(bg_image, caption="Generated Background", use_container_width=True)

st.success("Images saved in 'generated_images' folder")


