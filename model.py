import torch
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from google.colab import files
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

model_name = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

model = model.to(device)
model.eval()

print("CLIP model loaded successfully!")

historical_places = {
    "Taj Mahal": {
        "location": "Agra, Uttar Pradesh, India",
        "period": "17th century",
        "built": "1632–1653",
        "description": "A famous white marble mausoleum commissioned by Mughal emperor Shah Jahan.",
        "architecture": "Mughal architecture with Persian, Islamic and Indian influences"
    },

    "Red Fort": {
        "location": "Delhi, India",
        "period": "17th century",
        "built": "1639–1648",
        "description": "A historic Mughal fort constructed by Emperor Shah Jahan.",
        "architecture": "Mughal architecture"
    },

    "Qutub Minar": {
        "location": "Delhi, India",
        "period": "12th–13th century",
        "built": "1199 onwards",
        "description": "A monumental minaret and UNESCO World Heritage Site.",
        "architecture": "Indo-Islamic architecture"
    },

    "India Gate": {
        "location": "New Delhi, India",
        "period": "20th century",
        "built": "1921–1931",
        "description": "A war memorial dedicated to Indian soldiers who died in World War I.",
        "architecture": "Monumental arch architecture"
    },

    "Gateway of India": {
        "location": "Mumbai, Maharashtra, India",
        "period": "20th century",
        "built": "1911–1924",
        "description": "A monumental gateway overlooking the Arabian Sea in Mumbai.",
        "architecture": "Indo-Saracenic architecture"
    },

    "Mysore Palace": {
        "location": "Mysore, Karnataka, India",
        "period": "20th century",
        "built": "1897–1912",
        "description": "A grand palace and former royal residence of the Wadiyar dynasty.",
        "architecture": "Indo-Saracenic architecture"
    },

    "Hampi": {
        "location": "Vijayanagara, Karnataka, India",
        "period": "14th–16th century",
        "built": "Vijayanagara Empire period",
        "description": "An extensive archaeological site containing the remains of the Vijayanagara Empire.",
        "architecture": "Dravidian and Vijayanagara architecture"
    },

    "Ajanta Caves": {
        "location": "Maharashtra, India",
        "period": "2nd century BCE–6th century CE",
        "built": "Multiple construction periods",
        "description": "Rock-cut Buddhist caves famous for ancient paintings and sculptures.",
        "architecture": "Rock-cut Buddhist architecture"
    },

    "Ellora Caves": {
        "location": "Maharashtra, India",
        "period": "6th–10th century CE",
        "built": "Multiple construction periods",
        "description": "A major rock-cut complex containing Buddhist, Hindu and Jain monuments.",
        "architecture": "Rock-cut architecture"
    },

    "Konark Sun Temple": {
        "location": "Odisha, India",
        "period": "13th century",
        "built": "1250 CE",
        "description": "A monumental temple dedicated to the Hindu Sun God Surya.",
        "architecture": "Kalinga architecture"
    },

    "Charminar": {
        "location": "Hyderabad, Telangana, India",
        "period": "16th century",
        "built": "1591",
        "description": "A historic monument and mosque located in the heart of Hyderabad.",
        "architecture": "Indo-Islamic architecture"
    },

    "Sanchi Stupa": {
        "location": "Madhya Pradesh, India",
        "period": "3rd century BCE onwards",
        "built": "Commissioned under Emperor Ashoka",
        "description": "One of India's oldest surviving stone structures and an important Buddhist monument.",
        "architecture": "Buddhist stupa architecture"
    },

    "Mahabalipuram": {
        "location": "Tamil Nadu, India",
        "period": "7th–8th century CE",
        "built": "Pallava period",
        "description": "A historic coastal complex famous for temples and rock-cut monuments.",
        "architecture": "Pallava and Dravidian architecture"
    }
}

print("Number of historical places:", len(historical_places))

def recognize_historical_place(image, places):

    place_names = list(places.keys())

    # Create text descriptions for CLIP
    text_prompts = [
        f"a photograph of the historical monument {place}"
        for place in place_names
    ]

    # Process image and text
    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    # Move tensors to GPU/CPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Model prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Image-text similarity
    logits_per_image = outputs.logits_per_image

    probabilities = logits_per_image.softmax(dim=1)[0]

    # Top 5 predictions
    top_probs, top_indices = torch.topk(
        probabilities,
        min(5, len(place_names))
    )

    results = []

    for probability, index in zip(top_probs, top_indices):
        place = place_names[index.item()]

        results.append({
            "Place": place,
            "Confidence": float(probability.item() * 100)
        })

    return results

uploaded = files.upload()

filename = list(uploaded.keys())[0]

image = Image.open(filename).convert("RGB")

plt.figure(figsize=(10, 7))
plt.imshow(image)
plt.axis("off")
plt.title("Uploaded Historical Place")
plt.show()

results = recognize_historical_place(
    image,
    historical_places
)

results_df = pd.DataFrame(results)

print("\nTop Predictions:")
display(results_df)

best_prediction = results[0]

predicted_place = best_prediction["Place"]
confidence = best_prediction["Confidence"]

info = historical_places[predicted_place]

print("=" * 60)
print("HISTORICAL PLACE RECOGNITION RESULT")
print("=" * 60)

print(f"\nPredicted Place : {predicted_place}")
print(f"Confidence      : {confidence:.2f}%")

print("\n--- Historical Understanding ---")

print(f"Location        : {info['location']}")
print(f"Historical Era  : {info['period']}")
print(f"Construction    : {info['built']}")

print("\nDescription:")
print(info["description"])

print("\nArchitecture:")
print(info["architecture"])

print("=" * 60)

fig, axes = plt.subplots(
    1, 2,
    figsize=(16, 6)
)

# Original image
axes[0].imshow(image)
axes[0].axis("off")
axes[0].set_title(
    f"Predicted: {predicted_place}\n"
    f"Confidence: {confidence:.2f}%"
)

# Top predictions
names = [r["Place"] for r in results]
scores = [r["Confidence"] for r in results]

axes[1].barh(names[::-1], scores[::-1])
axes[1].set_xlabel("Confidence (%)")
axes[1].set_title("Top Historical Place Predictions")

plt.tight_layout()
plt.show()