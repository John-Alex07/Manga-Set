import os
import torch
import streamlit as st
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image
import glob
import base64

# Set cache directory
cache_dir = "J:\\Important\\VIT Vellore\\SET PROJECT\\cache"
torch.classes.__path__ = []  # Fix import issues

# Set background image for Streamlit app
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
        }}
        .stApp * {{
            text-shadow: 2px 2px 3px black, 0 0 1em black, 0 0 0.2em black;
            font-family: "Comic Sans MS", cursive, sans-serif;
            font-size: 18px;
            font-color: black;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

background_image_path = os.path.join("./", "Background.png")
if os.path.exists(background_image_path):
    set_background(background_image_path)

def load_model(model_name):
    """Load and cache the selected Stable Diffusion model."""
    scheduler = EulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
    model = StableDiffusionPipeline.from_pretrained(model_name, scheduler=scheduler, cache_dir=cache_dir, torch_dtype=torch.float16)
    model.to("cuda")
    print("CUDA Available: ", torch.cuda.is_available())
    return model

@st.cache_resource
def get_models():
    return {
        "Stable Diffusion 2": load_model("stabilityai/stable-diffusion-2"),
        "Stable Diffusion 2.1": load_model("stabilityai/stable-diffusion-2-1")
    }

models = get_models()

# Initialize session state variables
if "scene_name" not in st.session_state:
    st.session_state.scene_name = "Scene_1"

if "scene_seed" not in st.session_state:
    st.session_state.scene_seed = 42  # Default seed for consistency

if "mode" not in st.session_state:
    st.session_state.mode = "Character"

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

# Function to generate an image
def generate_image(prompt, negative_prompt, guidance_scale, num_inference_steps, save_path, model, seed):
    generator = torch.manual_seed(seed)  # Ensure consistent output
    image = model(prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, generator=generator).images[0]
    image.save(save_path)
    return image

# Streamlit UI
st.title("AI-Powered Scene & Character Generator")

# Model Selection
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Stable Diffusion 2"

model_choice = st.radio("Select Model:", ["Stable Diffusion 2", "Stable Diffusion 2.1"], index=0 if st.session_state.model_choice == "Stable Diffusion 2" else 1)
st.session_state.model_choice = model_choice
model = models[model_choice]

# Scene settings
scene_name = st.text_input("Enter scene name:", st.session_state.scene_name)
st.session_state.scene_name = scene_name

scene_seed = st.number_input("Set Scene Seed:", min_value=0, max_value=999999, value=st.session_state.scene_seed, key="scene_seed_input")
st.session_state.scene_seed = scene_seed

# Choose to generate Character, Background, or Scene
mode = st.radio("What would you like to generate?", ["Character", "Background", "Scene"], 
                index=["Character", "Background", "Scene"].index(st.session_state.mode))
st.session_state.mode = mode

# Common Prompt Inputs
prompt = st.text_area("Enter your prompt:", "A beautiful futuristic city at sunset")
negative_prompt = st.text_area("Negative prompt (optional):", "blurry, low-quality")
guidance_scale = st.slider("Guidance Scale", 1.0, 15.0, 7.5)
num_inference_steps = st.slider("Inference Steps", 10, 100, 50)

# Create folder for generated images
scene_path = os.path.join("generated_images", scene_name)
os.makedirs(scene_path, exist_ok=True)

# Character Generation
if mode == "Character":
    num_characters = st.number_input("Number of characters to generate:", 1, 10, 1)
    
    for i in range(num_characters):
        char_name = st.text_input(f"Character {i+1} Name:", f"Character_{i+1}", key=f"char_name_{i}")
        char_save_path = os.path.join(scene_path, f"{char_name}.png")

        if st.button(f"Generate {char_name}", key=f"gen_char_{i}"):
            with st.spinner(f"Generating {char_name}... Please wait."):
                char_image = generate_image(prompt, negative_prompt, guidance_scale, num_inference_steps, char_save_path, model, scene_seed)
            st.session_state.generated_images.append((char_name, char_image))
    # Display the generated character
    if st.session_state.generated_images:
        latest_char_name, latest_char_img = st.session_state.generated_images[-1]
        st.subheader(f"Latest Generated {latest_char_name}")
        st.image(latest_char_img, caption=latest_char_name, use_container_width=True)
        st.image(char_image, caption=char_name, use_container_width=True)

# Background Generation
elif mode == "Background":
    background_save_path = os.path.join(scene_path, "Background.png")
    
    if st.button("Generate Background"):
        with st.spinner("Generating background... Please wait."):
            bg_image = generate_image(prompt, negative_prompt, guidance_scale, num_inference_steps, background_save_path, model, scene_seed)
        st.session_state.generated_images.append(("Background", bg_image))
    # Display the generated background
    if st.session_state.generated_images:
        latest_bg_name, latest_bg_img = st.session_state.generated_images[-1]
        st.subheader(f"Latest Generated {latest_bg_name}")
        st.image(latest_bg_img, caption=latest_bg_name, use_container_width=True)
# Scene Generation (Full Scene)
elif mode == "Scene":
    scene_save_path = os.path.join(scene_path, "Full_Scene.png")

    if st.button("Generate Scene"):
        with st.spinner("Generating full scene... Please wait."):
            scene_image = generate_image(prompt, negative_prompt, guidance_scale, num_inference_steps, scene_save_path, model, scene_seed)
        st.session_state.generated_images.append(("Full Scene", scene_image))

    # Display the generated scene
    if os.path.exists(scene_save_path):
        st.subheader("Generated Scene")
        scene_img = Image.open(scene_save_path)
        st.image(scene_img, caption="Complete Scene", use_container_width=True)

# Sidebar Gallery for Selected Scene
st.sidebar.subheader("Generated Images Gallery")

# Select scene
scene_folders = glob.glob(os.path.join("generated_images", "Scene_*"))
selected_scene = st.sidebar.selectbox("Select Scene:", scene_folders, format_func=lambda x: os.path.basename(x))

if selected_scene:
    scene_folder = selected_scene

    # Show characters
    st.sidebar.subheader("Characters")
    character_images = glob.glob(os.path.join(scene_folder, "Character_*.png"))
    
    for char_path in character_images:
        char_name = os.path.basename(char_path).replace(".png", "")
        char_img = Image.open(char_path)
        st.sidebar.image(char_img, caption=char_name, use_container_width=True)

    # Show background
    st.sidebar.subheader("Background")
    background_path = os.path.join(scene_folder, "Background.png")
    
    if os.path.exists(background_path):
        background_img = Image.open(background_path)
        st.sidebar.image(background_img, caption="Background", use_container_width=True)

    # Show full scene
    st.sidebar.subheader("Full Scene")
    full_scene_path = os.path.join(scene_folder, "Full_Scene.png")

    if os.path.exists(full_scene_path):
        full_scene_img = Image.open(full_scene_path)
        st.sidebar.image(full_scene_img, caption="Full Scene", use_container_width=True)
