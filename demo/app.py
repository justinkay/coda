import os
os.environ['GRADIO_TEMP_DIR'] = "tmp/"

import gradio as gr
import json
import random
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import torch

from coda import CODA
from coda.datasets import Dataset
from coda.options import LOSS_FNS
from coda.oracle import Oracle


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

# Class names in order (0-5) from classes.txt
CLASS_NAMES = ["Jaguar", "Ocelot", "Mountain Lion", "Common Eland", "Waterbuck", "African Wild Dog"]
NAME_TO_CLASS_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Model information from models.txt
MODEL_INFO = [
    {"org": "Google", "name": "SigLIP2", "logo": "logos/google.png"},
    {"org": "OpenAI", "name": "CLIPViT-L", "logo": "logos/openai.png"},
    {"org": "Imageomics", "name": "BioCLIP", "logo": "logos/imageomics.png"}
]

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

# Load image filenames list
with open('images.txt', 'r') as f:
    image_filenames = [line.strip() for line in f.readlines() if line.strip()]

# Initialize CODA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = Dataset("iwildcam_demo.pt", device=device)
loss_fn = LOSS_FNS['acc']
oracle = Oracle(dataset, loss_fn=loss_fn)

# Create CODA selector with default parameters
class Args:
    def __init__(self):
        self.alpha = 0.9
        self.learning_rate = 0.01
        self.multiplier = 2.0
        self.prefilter_n = 0
        self.no_diag_prior = False
        self.q = "eig"
        self.method = "coda"
        self.loss = "acc"

args = Args()
coda_selector = CODA.from_args(dataset, args)

print(f"Initialized CODA with {dataset.preds.shape[1]} samples and {dataset.preds.shape[0]} models")

# Global state
current_image_info = None
# coda_selector already initialized above
# oracle already initialized above
# dataset already initialized above
# image_filenames already initialized above
iteration_count = 0

def get_model_predictions(chosen_idx):
    """Get model predictions and scores for a specific image"""
    global dataset

    if dataset is None or chosen_idx >= dataset.preds.shape[1]:
        return "No predictions available"

    # Get predictions for this image (shape: [num_models, num_classes])
    image_preds = dataset.preds[:, chosen_idx, :].detach().cpu().numpy()

    predictions_list = []

    for model_idx in range(image_preds.shape[0]):
        model_scores = image_preds[model_idx]
        predicted_class_idx = model_scores.argmax()
        predicted_class_name = CLASS_NAMES[predicted_class_idx]
        confidence = model_scores[predicted_class_idx]

        model_info = MODEL_INFO[model_idx]
        predictions_list.append(f"**{model_info['org']} {model_info['name']}:** {predicted_class_name} *({confidence:.3f})*")

    predictions_text = "### Model Predictions\n\n" + " | ".join(predictions_list)

    return predictions_text

def add_logo_to_x_axis(ax, x_pos, logo_path, model_name, height_px=35):
    """Add a logo image to x-axis next to model name"""
    try:
        img = mpimg.imread(logo_path)
        # Calculate zoom to achieve desired height in pixels
        # Rough conversion: height_px / image_height / dpi * 72
        zoom = height_px / img.shape[0] / ax.figure.dpi * 72
        imagebox = OffsetImage(img, zoom=zoom)

        # Position logo to the left of the x-tick
        logo_offset = -0.2  # Adjust this to move logo left/right relative to tick
        y_offset = -0.08
        ab = AnnotationBbox(imagebox, (x_pos + logo_offset, y_offset),
                           xycoords=('data', 'axes fraction'), frameon=False)
        ax.add_artist(ab)
    except Exception as e:
        print(f"Could not load logo {logo_path}: {e}")

def get_next_coda_image():
    """Get the next image that CODA wants labeled"""
    global current_image_info, coda_selector, iteration_count

    # Get next item from CODA
    chosen_idx, selection_prob = coda_selector.get_next_item_to_label()

    # Get the corresponding image filename
    if chosen_idx < len(image_filenames):
        filename = image_filenames[chosen_idx]
        image_path = os.path.join('iwildcam_demo_images', filename)

        # Find the corresponding annotation for this image
        current_image_info = None
        for annotation in data['annotations']:
            image_id = annotation['image_id']
            image_info = next((img for img in data['images'] if img['id'] == image_id), None)
            if image_info and image_info['file_name'] == filename:
                current_image_info = {
                    'filename': filename,
                    'species_id': annotation['category_id'],
                    'species_name': SPECIES_MAP[annotation['category_id']],
                    'chosen_idx': chosen_idx,
                    'selection_prob': selection_prob
                }
                break

        try:
            image = Image.open(image_path)
            predictions = get_model_predictions(chosen_idx)
            return image, f"Iteration {iteration_count}: CODA selected this image for labeling", predictions
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, f"Error loading image: {e}", "No predictions available"
    else:
        return None, "Image index out of range", "No predictions available"

def check_answer(user_choice):
    """Process user's label and update CODA"""
    global current_image_info, coda_selector, iteration_count

    if current_image_info is None:
        return "Please load an image first!", "", None, "No predictions", None, None

    correct_species = current_image_info['species_name']
    chosen_idx = current_image_info['chosen_idx']
    selection_prob = current_image_info['selection_prob']

    # Convert user choice to class index (0-5)
    if user_choice == "I don't know":
        # For "I don't know", we'll use the correct label but show it to user
        user_class_idx = NAME_TO_CLASS_IDX[correct_species]
        result = f"The correct answer was: {correct_species}"
    else:
        user_class_idx = NAME_TO_CLASS_IDX.get(user_choice, NAME_TO_CLASS_IDX[correct_species])
        if user_choice == correct_species:
            result = f"🎉 Correct! The last image was indeed a {correct_species}!"
        else:
            result = f"❌ Incorrect. The last image was a {correct_species}, not a {user_choice}. This may mislead the model selection process!"

    # Update CODA with the label
    coda_selector.add_label(chosen_idx, user_class_idx, selection_prob)
    iteration_count += 1

    # Get updated plots
    prob_plot = create_probability_chart()
    accuracy_plot = create_accuracy_chart()

    # Load next image
    next_image, status, predictions = get_next_coda_image()
    # Create HTML with inline help button for status
    status_html = f'{status} <span class="inline-help-btn" title="What is this?">?</span>'
    return result, status_html, next_image, predictions, prob_plot, accuracy_plot

def create_probability_chart():
    """Create a bar chart showing probability each model is best"""
    global coda_selector

    if coda_selector is None:
        # Fallback for initial state
        model_labels = [info['name'] for info in MODEL_INFO]
        probabilities = np.ones(len(MODEL_INFO)) / len(MODEL_INFO)  # Uniform prior
    else:
        probs_tensor = coda_selector.get_pbest()
        probabilities = probs_tensor.detach().cpu().numpy().flatten()
        model_labels = ["   " + info['name'] for info in MODEL_INFO[:len(probabilities)]]

    # Find the index of the highest probability
    best_idx = np.argmax(probabilities)

    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=150)

    # Create colors array - highlight the best model
    colors = ['orange' if i == best_idx else 'steelblue' for i in range(len(model_labels))]
    bars = ax.bar(range(len(model_labels)), probabilities, color=colors, alpha=0.7)

    # Add text above the highest bar
    ax.text(best_idx, probabilities[best_idx] + 0.0025, 'Current best guess',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Probability model is best', fontsize=12)
    ax.set_title(f'CODA Model Selection Probabilities (Iteration {iteration_count})', fontsize=12)
    ax.set_ylim(np.min(probabilities) - 0.01, np.max(probabilities) + 0.02)

    # Set x-axis labels and ticks
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, fontsize=12, ha='center')

    # Add logos to x-axis
    for i, model_info in enumerate(MODEL_INFO[:len(probabilities)]):
        add_logo_to_x_axis(ax, i, model_info['logo'], model_info['name'])
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save the figure and close it to prevent memory leaks
    temp_fig = fig
    plt.close(fig)
    return temp_fig

def create_accuracy_chart():
    """Create a bar chart showing true accuracy of each model"""
    global oracle, dataset

    if oracle is None or dataset is None:
        # Fallback for initial state
        model_labels = [info['name'] for info in MODEL_INFO]
        accuracies = np.random.random(len(MODEL_INFO))  # Random accuracies for now
    else:
        true_losses = oracle.true_losses(dataset.preds)
        # Convert losses to accuracies (assuming loss is 1 - accuracy)
        accuracies = (1 - true_losses).detach().cpu().numpy().flatten()
        model_labels = ["   " + info['name'] for info in MODEL_INFO[:len(accuracies)]]

    # Find the index of the highest accuracy
    best_idx = np.argmax(accuracies)

    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=150)

    # Create colors array - highlight the best model
    colors = ['red' if i == best_idx else 'forestgreen' for i in range(len(model_labels))]
    bars = ax.bar(range(len(model_labels)), accuracies, color=colors, alpha=0.7)

    # Add text above the highest bar
    ax.text(best_idx, accuracies[best_idx] + 0.005, 'True best model',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('True (oracle) \naccuracy of model', fontsize=12)
    ax.set_title('True Model Accuracies', fontsize=12)
    ax.set_ylim(np.min(accuracies) - 0.025, np.max(accuracies) + 0.05)

    # Set x-axis labels and ticks
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, fontsize=12, ha='center')

    # Add logos to x-axis
    for i, model_info in enumerate(MODEL_INFO[:len(accuracies)]):
        add_logo_to_x_axis(ax, i, model_info['logo'], model_info['name'])
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save the figure and close it to prevent memory leaks
    temp_fig = fig
    plt.close(fig)
    return temp_fig

# Create the Gradio interface
with gr.Blocks(title="CODA: Wildlife Photo Classification Challenge", 
               theme=gr.themes.Base(),
               css="""
               .subtle-outline {
                   border: 1px solid var(--border-color-primary) !important;
                   background: transparent !important;
                   border-radius: var(--radius-lg);
                   padding: 1rem;
               }
               .subtle-outline .flex {
                   background-color: white !important;
               }

               /* Popup overlay styles */
               .popup-overlay {
                   position: fixed;
                   top: 0;
                   left: 0;
                   width: 100%;
                   height: 100%;
                   background-color: rgba(0, 0, 0, 0.5);
                   z-index: 1000;
                   display: flex;
                   justify-content: center;
                   align-items: center;
               }

               .popup-overlay > div {
                   background: transparent !important;
                   border: none !important;
                   padding: 0 !important;
                   margin: 0 !important;
               }

               .popup-content {
                   background: white !important;
                   padding: 2rem !important;
                   border-radius: 1rem !important;
                   max-width: 850px;
                   width: 90%;
                   max-height: 80vh;
                   overflow-y: auto;
                   box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
                   border: none !important;
                   margin: 0 !important;
               }

               .popup-content > div {
                   background: white !important;
                   border: none !important;
                   padding: 0 !important;
                   margin: 0 !important;
               }

               /* Center title */
               .text-center {
                   text-align: center !important;
               }

               /* Subtitle styling */
               .subtitle {
                   text-align: center !important;
                   font-weight: 300 !important;
                   color: #666 !important;
                   margin-top: -0.5rem !important;
               }

               /* Question mark icon styling */
               .panel-container {
                   position: relative;
               }

               .help-icon {
                   position: absolute;
                   top: 5px;
                   right: 5px;
                   width: 25px;
                   height: 25px;
                   background-color: #f8f9fa;
                   color: #6c757d;
                   border: 1px solid #dee2e6;
                   border-radius: 50%;
                   display: flex;
                   align-items: center;
                   justify-content: center;
                   cursor: pointer;
                   font-size: 13px;
                   font-weight: 600;
                   z-index: 10;
                   transition: all 0.2s ease;
                   box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
               }

               .help-icon:hover {
                   background-color: #e9ecef;
                   color: #495057;
                   border-color: #adb5bd;
                   box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
               }

               /* Help popup styles */
               .help-popup-overlay {
                   position: fixed;
                   top: 0;
                   left: 0;
                   width: 100%;
                   height: 100%;
                   background-color: rgba(0, 0, 0, 0.5);
                   z-index: 1001;
                   display: flex;
                   justify-content: center;
                   align-items: center;
               }

               .help-popup-overlay > div {
                   background: transparent !important;
                   border: none !important;
                   padding: 0 !important;
                   margin: 0 !important;
               }

               .help-popup-content {
                   background: white !important;
                   padding: 1.5rem !important;
                   border-radius: 0.5rem !important;
                   max-width: 600px;
                   width: 90%;
                   box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
                   border: none !important;
                   margin: 0 !important;
               }

               .help-popup-content > div {
                   background: white !important;
                   border: none !important;
                   padding: 0 !important;
                   margin: 0 !important;
               }

               /* Inline help button */
               .inline-help-btn {
                   display: inline-block;
                   width: 20px;
                   height: 20px;
                   background-color: #f8f9fa;
                   color: #6c757d;
                   border: 1px solid #dee2e6;
                   border-radius: 50%;
                   text-align: center;
                   line-height: 18px;
                   cursor: pointer;
                   font-size: 11px;
                   font-weight: 600;
                   margin-left: 8px;
                   vertical-align: middle;
                   transition: all 0.2s ease;
                   box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
               }

               .inline-help-btn:hover {
                   background-color: #e9ecef;
                   color: #495057;
                   border-color: #adb5bd;
                   box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
               }

               #hidden-selection-help-btn {
                   display: none;
               }


               /* Reduce spacing around status text */
               .status-text {
                   margin: 0 !important;
                   padding: 0 !important;
               }

               .status-text > div {
                   margin: 0 !important;
                   padding: 0 !important;
               }

               /* Compact model predictions panel */
               .compact-predictions {
                   line-height: 1.1 !important;
                   margin: 0 !important;
                   padding: 0.1rem !important;
               }

               .compact-predictions p {
                   margin: 0.05rem 0 !important;
               }

               .compact-predictions h3 {
                   margin: 0 0 0.1rem 0 !important;
               }

               /* Target the subtle-outline group that contains predictions */
               .subtle-outline {
                   padding: 0.3rem !important;
                   margin: 0.2rem 0 !important;
               }

               /* Target the column inside the outline */
               .subtle-outline .flex {
                   padding: 0 !important;
                   margin: 0 !important;
               }
               """) as demo:
    # Main page title
    gr.Markdown("# CODA: Consensus-Driven Active Model Selection", elem_classes="text-center")

    # Popup component
    with gr.Group(visible=True, elem_classes="popup-overlay") as popup_overlay:
        with gr.Group(elem_classes="popup-content"):
            # Main intro content
            intro_content = gr.Markdown("""
            # CODA: Consensus-Driven Active Model Selection

            ## Wildlife Photo Classification Challenge

            You are a wildlife ecologist who has just collected a season's worth of imagery from cameras
            deployed in Africa and South America. You want to know what species occur in this imagery,
            and you hope to use a pre-trained classifier to give you answers quickly.
            But which one should you use?

            Instead of labeling a large validation set, our new method, **CODA**, enables you to perform **active model selection**.
            That is, CODA uses predictions from candidate models to guide the labeling process, querying you (a species identification expert)
            for labels on a select few images that will most efficiently differentiate between your candidate machine learning models.

            This demo lets you try CODA yourself! First, become a species identification expert by reading our classification guide
            so that you will be equipped to provide ground truth labels. Then, watch as CODA narrows down the best model over time
            as you provide labels for the query images. You will see that with your input CODA is able to identify the best model candidate
            with as few as ten (correctly) labeled images.
            """)

            # Species guide content (initially hidden)
            species_guide_content = gr.Markdown("""
            # Species Classification Guide

            Learn to identify the six wildlife species in this demo:

            ## Jaguar
            *Panthera onca*

            [Placeholder for jaguar image and description]

            ## Ocelot
            *Leopardus pardalis*

            [Placeholder for ocelot image and description]

            ## Mountain Lion
            *Puma concolor*

            [Placeholder for mountain lion image and description]

            ## Common Eland
            *Tragelaphus oryx*

            [Placeholder for common eland image and description]

            ## Waterbuck
            *Kobus ellipsiprymnus*

            [Placeholder for waterbuck image and description]

            ## African Wild Dog
            *Lycaon pictus*

            [Placeholder for african wild dog image and description]
            """, visible=False)

            gr.Markdown("<br>")  # Add some spacing

            with gr.Row():
                back_button = gr.Button("← Back to Intro", variant="secondary", size="lg", visible=False)
                guide_button = gr.Button("View Species Classification Guide", variant="secondary", size="lg")
                popup_start_button = gr.Button("Start Demo", variant="primary", size="lg")

    # Help popups for panels
    with gr.Group(visible=False, elem_classes="help-popup-overlay") as prob_help_popup:
        with gr.Group(elem_classes="help-popup-content"):
            gr.Markdown("""
            ## CODA Model Selection Probabilities

            This chart shows CODA's current confidence in each candidate model being the best performer.

            **How to read this chart:**
            - Each bar represents one of the three machine learning models
            - The height of each bar shows the probability (0-100%) that this model is the best
            - The orange bar indicates CODA's current best guess
            - As you provide more labels, CODA updates these probabilities

            **What you'll see:**
            - Initially, all models have similar probabilities (uniform prior)
            - As you label images, some models will gain confidence while others lose it
            - The goal is for one model to clearly emerge as the winner

            [Placeholder: This is where we would explain the specific methodology behind CODA's probability calculations]
            """)
            prob_help_close = gr.Button("Close", variant="secondary")

    with gr.Group(visible=False, elem_classes="help-popup-overlay") as acc_help_popup:
        with gr.Group(elem_classes="help-popup-content"):
            gr.Markdown("""
            ## True Model Accuracies

            This chart shows the actual performance of each model on the complete dataset (oracle knowledge).

            **How to read this chart:**
            - Each bar represents the true accuracy of one model
            - The red bar shows the actual best-performing model
            - This information is hidden from CODA during the selection process
            - You can compare this with CODA's estimates to see how well it's doing

            **Why this matters:**
            - This represents the "ground truth" that CODA is trying to discover
            - In real scenarios, you wouldn't know these true accuracies beforehand
            - The demo shows these to illustrate how CODA's estimates align with reality

            [Placeholder: This is where we would explain how these accuracies were computed and what they mean for the application]
            """)
            acc_help_close = gr.Button("Close", variant="secondary")

    with gr.Group(visible=False, elem_classes="help-popup-overlay") as selection_help_popup:
        with gr.Group(elem_classes="help-popup-content"):
            gr.Markdown("""
            ## How CODA Selects Images for Labeling

            This explains how CODA intelligently chooses which images to ask you to label next.

            **CODA's Selection Strategy:**
            - CODA uses **Expected Information Gain (EIG)** to select the most informative images
            - It chooses images where the candidate models disagree the most
            - Each selected image is expected to provide maximum information to distinguish between models
            - The selection process balances exploration of uncertain regions with exploitation of known patterns

            **Why This Image:**
            - This particular image was chosen because the models have conflicting predictions
            - Your label on this image will help CODA update its confidence in each model
            - The selection probability indicates how strongly CODA wanted to query this specific image
            - Images with higher disagreement among models are prioritized

            **Active Learning Benefits:**
            - This approach requires far fewer labels than random sampling
            - CODA can identify the best model with as few as 10-20 strategically chosen labels
            - The process is much more efficient than traditional validation approaches

            [Placeholder: This is where we would explain the mathematical details of the EIG calculation and model disagreement metrics]
            """)
            selection_help_close = gr.Button("Close", variant="secondary")
    
    # Two panels with bar charts
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="panel-container"):
                prob_help_button = gr.Button("?", elem_classes="help-icon", size="sm")
                prob_plot = gr.Plot(
                    value=None,
                    show_label=False
                )
        with gr.Column(scale=1):
            with gr.Group(elem_classes="panel-container"):
                acc_help_button = gr.Button("?", elem_classes="help-icon", size="sm")
                accuracy_plot = gr.Plot(
                    value=create_accuracy_chart(),
                    show_label=False
                )
    
    # Status display with help button
    status_with_help = gr.HTML("", visible=True, elem_classes="status-text")
    selection_help_button = gr.Button("", visible=False, elem_id="hidden-selection-help-btn")

    with gr.Row():
        image_display = gr.Image(
            label="Identify this animal:",
            value=None,
            height=400,
            width=550
        )

    # Model predictions panel (full width, single line)
    with gr.Group(elem_classes="subtle-outline"):
        with gr.Column(elem_classes="flex items-center justify-center h-full"):
            model_predictions_display = gr.Markdown(
                "### Model Predictions\n\n*Start the demo to see model votes!*",
                show_label=False,
                elem_classes="text-center compact-predictions"
            )
    
    gr.Markdown("### Which species is this?")

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
    
    # Add start over button
    start_over_button = gr.Button("Start Over", variant="secondary", size="lg")

    # Set up button interactions
    def start_demo():
        global iteration_count, coda_selector
        # Reset the demo state
        iteration_count = 0
        coda_selector = CODA.from_args(dataset, args)

        image, status, predictions = get_next_coda_image()
        prob_plot = create_probability_chart()
        acc_plot = create_accuracy_chart()
        # Create HTML with inline help button
        status_html = f'{status} <span class="inline-help-btn" title="What is this?">?</span>'
        return image, status_html, predictions, prob_plot, acc_plot, gr.update(visible=False), "", gr.update(visible=True)

    def start_over():
        global iteration_count, coda_selector
        # Reset the demo state
        iteration_count = 0
        coda_selector = CODA.from_args(dataset, args)

        # Reset all displays
        prob_plot = create_probability_chart()
        acc_plot = create_accuracy_chart()
        return None, "Demo reset. Click 'Start CODA Demo' to begin.", "### Model Predictions\n\n*Start the demo to see model votes!*", prob_plot, acc_plot, "", gr.update(visible=True), gr.update(visible=False)

    def show_species_guide():
        # Show species guide, hide intro content, show back button, hide guide button
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

    def show_intro():
        # Show intro content, hide species guide, hide back button, show guide button
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    def show_prob_help():
        return gr.update(visible=True)

    def hide_prob_help():
        return gr.update(visible=False)

    def show_acc_help():
        return gr.update(visible=True)

    def hide_acc_help():
        return gr.update(visible=False)

    def show_selection_help():
        return gr.update(visible=True)

    def hide_selection_help():
        return gr.update(visible=False)

    popup_start_button.click(
        fn=start_demo,
        outputs=[image_display, status_with_help, model_predictions_display, prob_plot, accuracy_plot, popup_overlay, result_display, selection_help_button]
    )

    start_over_button.click(
        fn=start_over,
        outputs=[image_display, status_with_help, model_predictions_display, prob_plot, accuracy_plot, result_display, popup_overlay, selection_help_button]
    )

    guide_button.click(
        fn=show_species_guide,
        outputs=[intro_content, species_guide_content, back_button, guide_button]
    )

    back_button.click(
        fn=show_intro,
        outputs=[intro_content, species_guide_content, back_button, guide_button]
    )

    # Help popup handlers
    prob_help_button.click(
        fn=show_prob_help,
        outputs=[prob_help_popup]
    )

    prob_help_close.click(
        fn=hide_prob_help,
        outputs=[prob_help_popup]
    )

    acc_help_button.click(
        fn=show_acc_help,
        outputs=[acc_help_popup]
    )

    acc_help_close.click(
        fn=hide_acc_help,
        outputs=[acc_help_popup]
    )

    selection_help_button.click(
        fn=show_selection_help,
        outputs=[selection_help_popup]
    )

    selection_help_close.click(
        fn=hide_selection_help,
        outputs=[selection_help_popup]
    )


    for btn in species_buttons:
        btn.click(
            fn=check_answer,
            inputs=[gr.State(btn.value)],
            outputs=[result_display, status_with_help, image_display, model_predictions_display, prob_plot, accuracy_plot]
        )

    idk_button.click(
        fn=check_answer,
        inputs=[gr.State("I don't know")],
        outputs=[result_display, status_with_help, image_display, model_predictions_display, prob_plot, accuracy_plot]
    )

    # Add JavaScript to handle inline help button clicks
    demo.load(
        lambda: None,
        outputs=[],
        js="""
        () => {
            setTimeout(() => {
                document.addEventListener('click', function(e) {
                    if (e.target && e.target.classList.contains('inline-help-btn')) {
                        e.preventDefault();
                        e.stopPropagation();
                        const hiddenBtn = document.getElementById('hidden-selection-help-btn');
                        if (hiddenBtn) {
                            hiddenBtn.click();
                        }
                    }
                });
            }, 100);
        }
        """
    )

if __name__ == "__main__":
    demo.launch(
        # share=True,
        server_port=7861
    )