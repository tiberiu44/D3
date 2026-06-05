import argparse
import json
import os

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
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
    parser.add_argument(
        "--crop-size",
        type=int,
        default=224,
        help="Crop size used by the classifier",
    )
    parser.add_argument(
        "--multi-crop",
        action="store_true",
        help="Run 5-crop inference (center + four corners) and mark fake if any crop is fake",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    if args.crop_size <= 0:
        parser.error("--crop-size must be a positive integer")
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


def get_multi_crops(image, crop_size):
    width, height = image.size
    if width < crop_size or height < crop_size:
        return [transforms.CenterCrop(crop_size)(image)]

    center_top = (height - crop_size) // 2
    center_left = (width - crop_size) // 2
    corners = [
        (center_top, center_left),  # center
        (0, 0),  # top-left
        (0, width - crop_size),  # top-right
        (height - crop_size, width - crop_size),  # bottom-right
        (height - crop_size, 0),  # bottom-left
    ]
    return [TF.crop(image, top, left, crop_size, crop_size) for top, left in corners]


def run_inference_batch(model, image_paths, transform, device, multi_crop=False, crop_size=224):
    tensors = []
    crop_counts = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if multi_crop:
                crops = get_multi_crops(image, crop_size)
            else:
                crops = [transforms.CenterCrop(crop_size)(image)]

            crop_tensors = [transform(crop) for crop in crops]
            tensors.extend(crop_tensors)
            crop_counts.append(len(crop_tensors))
    if not tensors:
        return []
    tensor = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        scores = model(tensor).sigmoid().view(-1).tolist()

    pred_labels = []
    offset = 0
    for crop_count in crop_counts:
        crop_scores = scores[offset : offset + crop_count]
        pred_labels.append(1 if any(score >= 0.5 for score in crop_scores) else 0)
        offset += crop_count
    return pred_labels


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
        pred_labels = run_inference_batch(
            model,
            batch_paths,
            transform,
            device,
            multi_crop=args.multi_crop,
            crop_size=args.crop_size,
        )

        for image_path, pred_label in zip(batch_paths, pred_labels):
            file_name = os.path.basename(image_path)
            explanation = FAKE_EXPLANATION if pred_label == 1 else REAL_EXPLANATION

            detection_rows.append({"id": file_name, "pred_label": pred_label})
            complex_rows.append({"id": file_name, "complex_explanation": explanation})
            simple_rows.append({"id": file_name, "simple_explanation": explanation})
            fake_count += pred_label

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
