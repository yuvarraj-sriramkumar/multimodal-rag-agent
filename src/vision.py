"""
vision.py — GPT-4o multimodal image analysis with chain-of-thought prompting.
"""

import base64
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a precise multimodal reasoning assistant.
When given an image, always respond in the following structure:

**Observation**
List what you directly see in the image — objects, colors, layout, text, etc.

**Reasoning**
Step through your analysis logically, connecting observations to draw conclusions.

**Answer**
Provide a concise, direct answer to the user's question."""


def encode_image(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image(
    image_path: str,
    question: str = "Describe what you see in this image.",
    context: str | None = None,
) -> str:
    """
    Send an image to GPT-4o with a chain-of-thought prompt.

    Args:
        image_path: Path to the local image file.
        question:   The user question about the image.
        context:    Optional extra context injected before the question.

    Returns:
        Model response structured as Observation / Reasoning / Answer.
    """
    client = OpenAI()

    image_b64 = encode_image(image_path)
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}
    mime_type = f"image/{mime_map.get(suffix, 'jpeg')}"

    user_content: list[dict] = []

    if context:
        user_content.append({"type": "text", "text": f"Context: {context}\n"})

    user_content.append({"type": "text", "text": question})
    user_content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_b64}", "detail": "high"},
        }
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1024,
        temperature=0.2,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from PIL import Image, ImageDraw

    # Create a synthetic test image locally (no network required)
    test_image = "/tmp/test_vision.png"
    img = Image.new("RGB", (300, 200), color=(70, 130, 180))  # steel blue background
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill=(255, 200, 0))   # yellow square
    draw.ellipse([160, 60, 270, 160], fill=(220, 50, 50))    # red circle
    draw.text((10, 170), "Multimodal RAG Test", fill=(255, 255, 255))
    img.save(test_image)
    print(f"Synthetic test image saved to {test_image}")

    print("Calling GPT-4o …\n")
    result = analyze_image(
        image_path=test_image,
        question="What shapes and colors do you see? What might this image represent?",
        context="This is a synthetic test image created programmatically for a multimodal RAG pipeline.",
    )
    print(result)
