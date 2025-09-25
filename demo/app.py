import os
os.environ['GRADIO_TEMP_DIR'] = "tmp/"


import gradio as gr
import json
import random
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

with open('iwildcam_demo_annotations.json', 'r') as f:
    data = json.load(f)

SPECIES_MAP = OrderedDict([
    (24, "Jaguar"),           # panthera onca
    (10, "Ocelot"),           # leopardus pardalis  
    (6, "Mountain Lion"),     # puma concolor
    (101, "Common Eland"),    # tragelaphus oryx
    (102, "Waterbuck"),       # kobus ellipsiprymnus
    (163, "African Wild Dog") # lycaon pictus
])
NAME_TO_ID = {name: id for id, name in SPECIES_MAP.items()}

# load image metadata
images_data = []
for annotation in tqdm(data['annotations'], desc='Loading annotations'):
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    image_info = next((img for img in data['images'] if img['id'] == image_id), None)
    if image_info:
        images_data.append({
            'filename': image_info['file_name'],
            'species_id': category_id,
            'species_name': SPECIES_MAP[category_id]
        })
print(f"Loaded {len(images_data)} images for the quiz")

# Global state
current_image_info = None

def get_random_image():
    """Select a random image and return it with the image object"""
    global current_image_info
    current_image_info = random.choice(images_data)
    
    image_path = os.path.join('iwildcam_demo_images', current_image_info['filename'])
    
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return get_random_image()  # Try another image

def check_answer(user_choice):
    """Check if the user's choice is correct"""
    if current_image_info is None:
        return "Please load an image first!", get_random_image()
    
    correct_species = current_image_info['species_name']
    
    if user_choice == "I don't know":
        result = f"The correct answer was: {correct_species}"
    elif user_choice == correct_species:
        result = f"🎉 Correct! This is indeed a {correct_species}!"
    else:
        result = f"❌ Incorrect. This is a {correct_species}, not a {user_choice}."
    
    # Load next image
    next_image = get_random_image()
    return result, next_image

# Create the Gradio interface
with gr.Blocks(title="Wildlife Photo Classification Challenge", 
               theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🦁 Wildlife Photo Classification Challenge
    
    Test your knowledge of African and South American wildlife! 
    Look at each photo and try to identify the species.
    
    **Instructions:**
    - Look carefully at the image
    - Click the button for the species you think it is
    - Click "I don't know" if you're not sure
    - You'll get feedback on whether you were correct, then move to the next image
    """)
    
    with gr.Row():
        image_display = gr.Image(
            label="Identify this animal:", 
            value=get_random_image(),
            height=500,
            width=700
        )
    
    gr.Markdown("### Which species do you think this is?")
    
    with gr.Row():
        # Create buttons for each species
        species_buttons = []
        for species_name in SPECIES_MAP.values():
            btn = gr.Button(species_name, variant="secondary", size="lg")
            species_buttons.append(btn)
        
        # Add "I don't know" button
        idk_button = gr.Button("I don't know", variant="primary", size="lg")
    
    # Result display
    result_display = gr.Markdown("", visible=True)
    
    # Set up button interactions
    for btn in species_buttons:
        btn.click(
            fn=check_answer,
            inputs=[gr.State(btn.value)],
            outputs=[result_display, image_display]
        )
    
    idk_button.click(
        fn=check_answer,
        inputs=[gr.State("I don't know")],
        outputs=[result_display, image_display]
    )

if __name__ == "__main__":
    demo.launch(
        # share=True,
        server_port=7861
    )