#!/usr/bin/env python3
"""
Zero-shot inference script for demo images using Hugging Face models.
Runs inference on images in demo/iwildcam_demo_images using specified models
and saves results to JSON files.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

# Species mapping from demo/app.py
SPECIES_MAP = OrderedDict([
    (24, "Jaguar"),           # panthera onca
    (10, "Ocelot"),           # leopardus pardalis
    (6, "Mountain Lion"),     # puma concolor
    (101, "Common Eland"),    # tragelaphus oryx
    (102, "Waterbuck"),       # kobus ellipsiprymnus
    (163, "African Wild Dog") # lycaon pictus
])

# Class names
CLASS_NAMES = list(SPECIES_MAP.values())

# More descriptive class names for better zero-shot performance
DESCRIPTIVE_CLASS_NAMES = [
    "a jaguar cat",
    "an ocelot cat",
    "a mountain lion cougar",
    "a common eland antelope",
    "a waterbuck antelope",
    "an African wild dog"
]

# Models to test
MODELS = [
    "openai/clip-vit-large-patch14",
    "google/siglip2-so400m-patch16-naflex",
    "imageomics/bioclip"
]

def load_demo_annotations():
    """Load the demo annotations to get image metadata."""
    with open('iwildcam_demo_annotations.json', 'r') as f:
        data = json.load(f)

    # Create mapping from filename to metadata
    image_metadata = {}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        image_info = next((img for img in data['images'] if img['id'] == image_id), None)
        if image_info:
            image_metadata[image_info['file_name']] = {
                'species_id': category_id,
                'species_name': SPECIES_MAP.get(category_id, "Unknown")
            }

    return image_metadata

def run_bioclip_inference(image_paths, class_names):
    """Run zero-shot inference using BioCLIP with pybioclip."""
    try:
        from bioclip import CustomLabelsClassifier
        print("Loading BioCLIP model...")

        # Create classifier with custom labels
        device = "cuda" if torch.cuda.is_available() else "cpu"
        classifier = CustomLabelsClassifier(
            cls_ary=class_names,
            device=device
        )

        results = {}

        for i, image_path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")

            try:
                # Use classifier's predict function
                predictions = classifier.predict(image_path, k=len(class_names))

                # Convert predictions to our format
                scores = {}

                # Initialize all classes with 0
                for class_name in class_names:
                    scores[class_name] = 0.0

                # Fill in the predictions - predictions is a list of dicts with format:
                # [{'file_name': '...', 'classification': 'Ocelot', 'score': 0.999}, ...]
                for pred in predictions:
                    class_name = pred['classification']
                    score = pred['score']
                    if class_name in scores:
                        scores[class_name] = score

                results[os.path.basename(image_path)] = scores

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                # Fill with uniform probabilities if processing fails
                uniform_prob = 1.0 / len(class_names)
                results[os.path.basename(image_path)] = {class_name: uniform_prob for class_name in class_names}

        return results

    except ImportError:
        print("bioclip not available, skipping BioCLIP")
        return None
    except Exception as e:
        print(f"Error loading BioCLIP: {e}")
        return None

def run_siglip_inference(image_paths, class_names):
    """Run zero-shot inference using SigLIP with manual CLIP-style computation."""
    print("Loading SigLIP model...")
    try:
        from transformers import AutoProcessor, AutoModel

        model_name = "google/siglip2-so400m-patch16-naflex"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        results = {}

        with torch.no_grad():
            for i, image_path in enumerate(image_paths):
                if i % 10 == 0:
                    print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")

                try:
                    # Load and process image
                    image = Image.open(image_path).convert("RGB")

                    # Create prompt templates for SigLIP (as recommended in docs)
                    prompts = [f"This is a photo of a {class_name.lower()}" for class_name in class_names]

                    # Process inputs with proper padding
                    inputs = processor(
                        text=prompts,
                        images=image,
                        return_tensors="pt",
                        padding="max_length"
                    ).to(device)

                    # Get outputs
                    outputs = model(**inputs)

                    # Get similarity scores (logits_per_image)
                    logits_per_image = outputs.logits_per_image

                    # Apply sigmoid to get probabilities (SigLIP uses pairwise sigmoid loss)
                    sigmoid_probs = torch.sigmoid(logits_per_image).squeeze(0)

                    # Convert to relative probabilities using softmax for comparison
                    probabilities = torch.softmax(logits_per_image, dim=-1).squeeze(0)

                    # Convert to dictionary
                    scores = {}
                    for j, class_name in enumerate(class_names):
                        scores[class_name] = probabilities[j].item()

                    results[os.path.basename(image_path)] = scores

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    results[os.path.basename(image_path)] = {class_name: 0.0 for class_name in class_names}

        return results

    except Exception as e:
        print(f"Error loading SigLIP: {e}")
        return None

def run_zeroshot_inference(model_name, image_paths, class_names):
    """Run zero-shot inference using specified model."""
    print(f"Loading model: {model_name}")

    try:
        # Create zero-shot image classification pipeline
        classifier = pipeline(
            "zero-shot-image-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )

        results = {}

        for i, image_path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")

            try:
                # Load and process image
                image = Image.open(image_path).convert("RGB")

                # Run inference
                outputs = classifier(image, class_names)

                # Convert to post-softmax scores
                scores = {}
                for output in outputs:
                    scores[output['label']] = output['score']

                # Ensure all classes are present (fill missing with 0)
                for class_name in class_names:
                    if class_name not in scores:
                        scores[class_name] = 0.0

                results[os.path.basename(image_path)] = scores

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                # Fill with zeros if processing fails
                results[os.path.basename(image_path)] = {class_name: 0.0 for class_name in class_names}

        return results

    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

def main():
    """Main function to run zero-shot inference on all models."""
    # Get list of demo images
    image_dir = "iwildcam_demo_images"
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_paths = [os.path.join(image_dir, f) for f in image_files]

    print(f"Found {len(image_files)} demo images")

    # Load annotations for reference
    image_metadata = load_demo_annotations()
    print(f"Loaded metadata for {len(image_metadata)} images")

    # Run inference for each model
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Running inference with {model_name}")
        print(f"{'='*60}")

        # Check if results already exist
        model_safe_name = model_name.replace("/", "_").replace("-", "_")
        output_file = f"zeroshot_results_{model_safe_name}.json"

        if os.path.exists(output_file):
            print(f"Results file {output_file} already exists, skipping {model_name}")
            continue

        # Handle different models with appropriate methods
        if model_name == "imageomics/bioclip":
            results = run_bioclip_inference(image_paths, CLASS_NAMES)
        elif model_name == "google/siglip2-so400m-patch16-naflex":
            results = run_siglip_inference(image_paths, CLASS_NAMES)
        else:
            results = run_zeroshot_inference(model_name, image_paths, CLASS_NAMES)

        if results is not None:
            # Add metadata to results
            output_data = {
                "model": model_name,
                "class_names": CLASS_NAMES,
                "num_images": len(results),
                "results": results
            }

            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"Results saved to {output_file}")

            # Print sample results
            sample_images = list(results.keys())[:3]
            print(f"\nSample results from {model_name}:")
            for img in sample_images:
                print(f"  {img}:")
                scores = results[img]
                # Show top 3 predictions
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for class_name, score in sorted_scores[:3]:
                    print(f"    {class_name}: {score:.4f}")
        else:
            print(f"Failed to run inference with {model_name}")

if __name__ == "__main__":
    # Change to demo directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()