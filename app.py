import streamlit as st
from deep_translator import GoogleTranslator
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Translation function using deep-translator (fixed to English)
def get_translation(text):
    translated_text = GoogleTranslator(source='auto', target='en').translate(text)
    return translated_text

# Configuration for image generation
class CFG:
    device = "cpu"  # Use CPU instead of GPU
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (512, 512)  # Resized image size for display
    image_gen_guidance_scale = 9

# Load the image generation model
@st.cache_resource
@st.cache_resource
def load_image_gen_model():
    api_key = "hf_grUtAnxhf_grUtAnxhKanTuCIVQwWnLWsMHfmgxYCbRqhKanTuCIVQwWnLWsMHfmgxYCbRq"  # Replace with your key
    model = StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id,
        torch_dtype=torch.float32,
        use_auth_token=api_key  # Add API key here
    )
    model = model.to(CFG.device)
    return model


# Function to generate image
def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    image = image.resize(CFG.image_gen_size)
    return image

# Streamlit interface
st.title("Mulit-Lingual text to Image Generator")
st.write("Generate an image based on your input prompt!")

# User input for prompt
user_prompt = st.text_input("Enter your prompt:", "A dog with a human body sitting on the iron throne")

# Translate prompt
if user_prompt:
    translation = get_translation(user_prompt)
    st.write(f"Translated Prompt: {translation}")

    # Button to trigger image generation
    if st.button("Generate Image"):
        # Load the model
        image_gen_model = load_image_gen_model()

        # Generate image
        with st.spinner('Generating image...'):
            generated_image = generate_image(translation, image_gen_model)

        # Display the image
        st.image(generated_image, caption="Generated Image", use_column_width=True)


