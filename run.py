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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of images to process per inference batch",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    return args


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


def run_inference_batch(model, image_paths, transform, device):
    tensors = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            tensors.append(transform(image.convert("RGB")))
    if not tensors:
        return []
    tensor = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        scores = model(tensor).sigmoid().view(-1).tolist()
    return [1 if score >= 0.5 else 0 for score in scores]


def write_jsonl_lines(output_path, rows):
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
    image_paths = get_image_list(args.input_folder)
    total_files = len(image_paths)
    print(f"Found {total_files} image files in {args.input_folder}")
    print(f"Running inference with batch size: {args.batch_size}")

    fake_count = 0
    processed_count = 0
    for batch_start in range(0, total_files, args.batch_size):
        batch_paths = image_paths[batch_start : batch_start + args.batch_size]
        pred_labels = run_inference_batch(model, batch_paths, transform, device)

        for image_path, pred_label in zip(batch_paths, pred_labels):
            file_name = os.path.basename(image_path)
            explanation = FAKE_EXPLANATION if pred_label == 1 else REAL_EXPLANATION

            detection_rows.append({"id": file_name, "pred_label": pred_label})
            complex_rows.append({"id": file_name, "complex_explanation": explanation})
            simple_rows.append({"id": file_name, "simple_explanation": explanation})
            fake_count += int(pred_label == 1)

        processed_count += len(batch_paths)
        print(f"Processed {processed_count}/{total_files} files")

    detection_output = os.path.join(args.output_folder, "detection.jsonl")
    complex_output = os.path.join(args.output_folder, "complex.jsonl")
    simple_output = os.path.join(args.output_folder, "simple.jsonl")
    write_jsonl_lines(detection_output, detection_rows)
    write_jsonl_lines(complex_output, complex_rows)
    write_jsonl_lines(simple_output, simple_rows)

    real_count = total_files - fake_count
    print(
        "Inference complete. "
        f"Total files: {total_files}, fake predictions: {fake_count}, real predictions: {real_count}"
    )
    print(f"Wrote detection output to: {detection_output}")
    print(f"Wrote complex explanations to: {complex_output}")
    print(f"Wrote simple explanations to: {simple_output}")


if __name__ == "__main__":
    main()
