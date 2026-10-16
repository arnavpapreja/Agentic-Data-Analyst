from __future__ import annotations
import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps, ImageEnhance
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
TASK_INSTRUCTIONS = r"""
You are transcribing three consecutive English manga pages. Return exactly one valid JSON object and no markdown.
Required schema:
{"sequence_id":"TARGET_SEQUENCE_ID","pages":[[{"speaker":"char1","text":"..."}],[],[]]}
Rules:
- The pages are ordered PAGE 1, PAGE 2, PAGE 3. Return exactly three page lists.
- Include dialogue, internal thoughts, narrator text, clear spoken screams/grunts/vocalisations, and punctuation-only speech/thought balloons.
- Exclude visual sound effects and translated sound-effect captions, tiny unclear breath/reaction text, titles/logos/credits/page numbers/ads/watermarks, character introduction labels, editor/scanlator notes, writing on signs/clothes/objects, and content inside letters, diaries, phone messages, news pages, documents or interfaces.
- Follow the intended visual order. Manga is usually read right-to-left, but follow panel layout and balloon connections when layout requires a different order. Preserve stutters, meaningful repetition, and punctuation. Do not split one utterance just because the printed words wrap onto multiple lines.
- Each object represents one utterance/thought/narration item, not every printed line of the same balloon.
- Use the same speaker label for the same character across all three pages. The real name is unnecessary; use char1, char2, etc. consistently. Different sequences may reuse labels.
- Use speaker "NARRATION" only for narrator text. If the speaker genuinely cannot be determined, use "UNKNOWN".
- Do not invent text. If a page has no included story text, use an empty list.
- Make a best-effort transcription; do not describe images or explain your choices.
""".strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_paths(sequence_id: str, split: str) -> list[Path]:
    folder = DATA / split / "images" / sequence_id
    paths = [folder / f"{i:02d}.png" for i in (1, 2, 3)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image(s) for {sequence_id}: {missing}")
    return paths


def load_page(path: Path, max_edge: int = 1600) -> Image.Image:
    image = Image.open(path).convert("RGB")
    # Mild contrast normalization, while preserving the original page layout.
    image = ImageOps.autocontrast(image.convert("L")).convert("RGB")
    w, h = image.size
    edge = max(w, h)
    if edge > max_edge:
        scale = max_edge / edge
        image = image.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    return image


def get_images(sequence_id: str, split: str) -> list[Image.Image]:
    return [load_page(path) for path in image_paths(sequence_id, split)]


def sequence_content() -> list[dict[str, str]]:
    # Image markers are matched to the flat images list in the same order.
    content: list[dict[str, str]] = []
    for page_no in range(1, 4):
        content.append({"type": "text", "text": f"PAGE {page_no}:"})
        content.append({"type": "image"})
    return content


def choose_shots(pool: list[dict[str, Any]], target_id: str, count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0 or not pool:
        return []
    candidates = [row for row in pool if row["sequence_id"] != target_id]
    if not candidates:
        return []
    # Deterministic sampling: repeated runs use the same labelled demonstrations.
    rng = random.Random(f"{seed}:{target_id}")
    if count >= len(candidates):
        return candidates
    return rng.sample(candidates, count)


def build_prompt(processor, target_id: str, target_images: list[Image.Image], shots: list[dict[str, Any]], split: str):
    messages: list[dict[str, Any]] = []
    all_images: list[Image.Image] = []
    for shot in shots:
        shot_images = get_images(shot["sequence_id"], "development")
        content = [{"type": "text", "text": "Here is a labelled example sequence. Learn the required transcription style and consistent speaker labelling. Speaker labels only apply inside this example."}]
        content.extend(sequence_content())
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps(shot, ensure_ascii=False)}]})
        all_images.extend(shot_images)
    target_content: list[dict[str, str]] = [{"type": "text", "text": f"Now solve the target sequence with sequence_id={target_id}."}]
    target_content.extend(sequence_content())
    target_content.append({"type": "text", "text": TASK_INSTRUCTIONS.replace("TARGET_SEQUENCE_ID", target_id)})
    messages.append({"role": "user", "content": target_content})
    all_images.extend(target_images)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    return prompt, all_images


def extract_json(raw: str) -> dict[str, Any] | None:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # First try the full output, then locate the first decodable object.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned[match.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def normalize_prediction(obj: dict[str, Any], sequence_id: str) -> dict[str, Any] | None:
    pages = obj.get("pages")
    if not isinstance(pages, list) or len(pages) != 3:
        return None
    normalized_pages = []
    for page in pages:
        if not isinstance(page, list):
            return None
        clean_page = []
        for item in page:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            speaker = item.get("speaker", "UNKNOWN")
            if not isinstance(text, str):
                continue
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            if not isinstance(speaker, str) or not speaker.strip():
                speaker = "UNKNOWN"
            speaker = speaker.strip()
            clean_page.append({"speaker": speaker, "text": text})
        normalized_pages.append(clean_page)
    return {"sequence_id": sequence_id, "pages": normalized_pages}


def tesseract_fallback(image: Image.Image) -> list[dict[str, str]]:
    """Automatic emergency fallback if the VLM returns invalid JSON twice."""
    exe = shutil.which("tesseract")
    if not exe:
        return []
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "page.png"
        image.save(img_path)
        run = subprocess.run([exe, str(img_path), "stdout", "--psm", "11"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = []
    for line in run.stdout.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if sum(ch.isalpha() for ch in line) >= 2:
            lines.append({"speaker": "UNKNOWN", "text": line})
    return lines


def generate_one(model, processor, sequence_id: str, split: str, shots: list[dict[str, Any]], max_new_tokens: int) -> tuple[dict[str, Any], str]:
    target_images = get_images(sequence_id, split)
    prompt, all_images = build_prompt(processor, sequence_id, target_images, shots, split)
    device = next(model.parameters()).device
    inputs = processor(text=prompt, images=all_images, return_tensors="pt")
    inputs = inputs.to(device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.03)
    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    normalized = normalize_prediction(extract_json(raw) or {}, sequence_id)
    if normalized is not None:
        return normalized, raw
    # Retry with a stricter output repair instruction, preserving no external service use.
    repair_prompt = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": "Convert the following attempted transcription into valid JSON with exactly keys sequence_id and pages; pages must be three arrays of objects with exactly speaker and text. Keep the content and do not add anything. sequence_id must be " + json.dumps(sequence_id) + "."}]},
        {"role": "assistant", "content": [{"type": "text", "text": raw}]},
        {"role": "user", "content": [{"type": "text", "text": "Output only the corrected JSON object now."}]},
    ], add_generation_prompt=True)
    repair_inputs = processor(text=repair_prompt, return_tensors="pt").to(device)
    repair_ids = model.generate(**repair_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    repair_raw = processor.batch_decode(repair_ids[:, repair_inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    normalized = normalize_prediction(extract_json(repair_raw) or {}, sequence_id)
    if normalized is not None:
        return normalized, raw + "\n[REPAIRED]\n" + repair_raw
    pages = [tesseract_fallback(img) for img in target_images]
    return {"sequence_id": sequence_id, "pages": pages}, raw + "\n[OCR FALLBACK USED]"


def build_model(model_name: str):
    if not torch.cuda.is_available():
        print("WARNING: CUDA was not detected; local CPU inference may be very slow.", flush=True)
        dtype = torch.float32
        device = "cpu"
    else:
        dtype = torch.float16
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=dtype,
        _attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, processor


def main():
    parser = argparse.ArgumentParser(description="Run open-weight local VLM inference for the Manga Task.")
    parser.add_argument("--split", choices=["development", "test"], required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--shots", type=int, default=1, help="Number of labelled development sequences used as visual in-context examples (0 is zero-shot baseline).")
    parser.add_argument("--shot-pool", type=Path, default=DATA / "development" / "labels.jsonl")
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional file with one sequence_id per line, for validation runs.")
    parser.add_argument("--max-new-tokens", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.shots < 0 or args.shots > 2:
        raise SystemExit("For memory safety, --shots must be 0, 1, or 2.")
    output = args.output or (ROOT / "outputs" / ("test_predictions.jsonl" if args.split == "test" else "development_predictions.jsonl"))
    if args.split == "test":
        targets = [r["sequence_id"] for r in load_jsonl(DATA / "sample_submission.jsonl")]
    else:
        labels = load_jsonl(DATA / "development" / "labels.jsonl")
        targets = [r["sequence_id"] for r in labels]
    if args.ids_file:
        requested = [line.strip() for line in args.ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        allowed = set(targets)
        invalid = [sid for sid in requested if sid not in allowed]
        if invalid:
            raise SystemExit(f"Unknown {args.split} sequence id(s): {invalid[:5]}")
        targets = requested
    if args.limit is not None:
        targets = targets[:max(0, args.limit)]
    pool = load_jsonl(args.shot_pool) if args.shots else []
    # In validation mode, avoid using the target sequence itself as a demonstration.
    model, processor = build_model(args.model)
    results = []
    log_path = ROOT / "logs" / (output.stem + "_raw.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for index, sequence_id in enumerate(targets, 1):
            shots = choose_shots(pool, sequence_id, args.shots, args.seed)
            print(f"[{index}/{len(targets)}] {sequence_id} | shots={len(shots)}", flush=True)
            prediction, raw = generate_one(model, processor, sequence_id, args.split, shots, args.max_new_tokens)
            results.append(prediction)
            log.write(f"### {sequence_id}\n{raw}\n\n")
            log.flush()
            write_jsonl(output, results)
    write_jsonl(output, results)
    print(f"Wrote {len(results)} predictions to {output}")
    print(f"Raw model outputs/log: {log_path}")

if __name__ == "__main__":
    main()
