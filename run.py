import argparse
import json
import os

import torch
import torchvision.transforms as transforms
from PIL import Image

from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer


FAKE_EXPLANATION = (
    "Analyzing the color scheme, shadows, lighting, and fine-level details, "
    "there is indication that this file has been tampered with."
)
REAL_EXPLANATION = "This image appears legitimate"

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", required=True, help="Path to model .pth file")
    parser.add_argument("--input_folder", required=True, help="Folder with input .png/.webp images")
    parser.add_argument("--output_folder", required=True, help="Folder where JSONL files are written")
    return parser.parse_args()


def get_image_list(input_folder):
    valid_ext = {".png", ".webp"}
    image_files = []
    for file_name in os.listdir(input_folder):
        full_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(full_path):
            continue
        ext = os.path.splitext(file_name)[1].lower()
        if ext in valid_ext:
            image_files.append(full_path)
    return sorted(image_files)


def build_model(model_path, device):
    model = CLIPModelShuffleAttentionPenultimateLayer(
        "ViT-L/14", shuffle_times=1, original_times=1, patch_size=[14]
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.attention_head.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def run_inference(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(tensor).sigmoid().item()
    return 1 if score >= 0.5 else 0


def write_jsonl_lines(output_path, rows):
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_path, device)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    detection_rows = []
    complex_rows = []
    simple_rows = []

    for image_path in get_image_list(args.input_folder):
        file_name = os.path.basename(image_path)
        pred_label = run_inference(model, image_path, transform, device)
        explanation = FAKE_EXPLANATION if pred_label == 1 else REAL_EXPLANATION

        detection_rows.append({"id": file_name, "pred_label": pred_label})
        complex_rows.append({"id": file_name, "complex_explanation": explanation})
        simple_rows.append({"id": file_name, "simple_explanation": explanation})

    write_jsonl_lines(os.path.join(args.output_folder, "detection.jsonl"), detection_rows)
    write_jsonl_lines(os.path.join(args.output_folder, "complex.jsonl"), complex_rows)
    write_jsonl_lines(os.path.join(args.output_folder, "simple.jsonl"), simple_rows)


if __name__ == "__main__":
    main()
