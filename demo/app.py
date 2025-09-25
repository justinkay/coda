import os
os.environ['GRADIO_TEMP_DIR'] = "tmp/"


import gradio as gr
import json
import random
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

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

def create_probability_chart():
    """Create a bar chart showing probability each model is best"""
    models = [f"Model {i}" for i in range(1, 11)]
    probabilities = np.random.random(10)  # Random probabilities for now
    
    # Find the index of the highest probability
    best_idx = np.argmax(probabilities)
    
    fig, ax = plt.subplots(figsize=(6, 2.5))
    
    # Create colors array - highlight the best model
    colors = ['orange' if i == best_idx else 'steelblue' for i in range(len(models))]
    bars = ax.bar(models, probabilities, color=colors, alpha=0.7)
    
    # Add text above the highest bar
    ax.text(best_idx, probabilities[best_idx] + 0.05, 'Current best guess', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Probability model is best', fontsize=10)
    ax.set_xlabel('Models', fontsize=10)
    ax.set_title('Model Selection Probabilities', fontsize=11)
    ax.set_ylim(0, 1.2)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    return fig

def create_accuracy_chart():
    """Create a bar chart showing true accuracy of each model"""
    models = [f"Model {i}" for i in range(1, 11)]
    accuracies = np.random.random(10)  # Random accuracies for now
    
    # Find the index of the highest accuracy
    best_idx = np.argmax(accuracies)
    
    fig, ax = plt.subplots(figsize=(6, 2.5))
    
    # Create colors array - highlight the best model
    colors = ['red' if i == best_idx else 'forestgreen' for i in range(len(models))]
    bars = ax.bar(models, accuracies, color=colors, alpha=0.7)
    
    # Add text above the highest bar
    ax.text(best_idx, accuracies[best_idx] + 0.05, 'True best model', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('True (oracle) \naccuracy of model', fontsize=10)
    ax.set_xlabel('Models', fontsize=10)
    ax.set_title('True Model Accuracies', fontsize=11)
    ax.set_ylim(0, 1.2)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    return fig

# Create the Gradio interface
with gr.Blocks(title="CODA: Wildlife Photo Classification Challenge", 
               theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Consensus-Driven Active Model Selection: Wildlife Photo Classification Challenge")
    
    # Two panels with bar charts
    with gr.Row():
        with gr.Column(scale=1):
            prob_plot = gr.Plot(
                value=create_probability_chart(),
                show_label=False
            )
        with gr.Column(scale=1):
            accuracy_plot = gr.Plot(
                value=create_accuracy_chart(),
                show_label=False
            )
    
    with gr.Row():
        image_display = gr.Image(
            label="Identify this animal:", 
            value=get_random_image(),
            height=400,
            width=550
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