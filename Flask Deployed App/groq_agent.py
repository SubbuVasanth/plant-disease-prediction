"""
groq_agent.py — Agentic LLM layer using Groq (LLaMA 3.3-70B Versatile)

Takes the CNN-predicted disease class and confidence score, sends a
structured prompt to the Groq LLM, and returns actionable agricultural
advice aligned with UN SDG 2 (Zero Hunger).
"""

import os
import json
from groq import Groq

# ---------------------------------------------------------------------------
# Groq client — reads GROQ_API_KEY from the environment
# ---------------------------------------------------------------------------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"


def get_diagnosis(predicted_class: str, confidence: float) -> dict:
    """
    Call the Groq LLM with the CNN prediction and return a structured
    JSON object containing:
      - disease_name
      - diagnosis
      - treatment_steps  (list)
      - prevention_tips  (list)
      - sdg2_impact      (string — UN SDG 2 Zero Hunger relevance)

    Parameters
    ----------
    predicted_class : str
        The class label predicted by the CNN (e.g. "Tomato___Late_blight").
    confidence : float
        The softmax confidence score (0-1) for the predicted class.

    Returns
    -------
    dict
        Parsed JSON with diagnosis details, or a fallback error dict.
    """

    prompt = f"""You are AgriScan AI, an expert agricultural disease advisor.

A Convolutional Neural Network analysed a plant leaf image and returned:
  • Predicted class : {predicted_class}
  • Confidence      : {confidence:.2%}

Based on this prediction, provide the following information in **valid JSON only** (no markdown, no extra text):

{{
  "disease_name"    : "<human-readable disease name>",
  "diagnosis"       : "<one-paragraph diagnosis explanation>",
  "treatment_steps" : ["<step 1>", "<step 2>", "<step 3>"],
  "prevention_tips" : ["<tip 1>", "<tip 2>", "<tip 3>"],
  "sdg2_impact"     : "<explain how early detection of this disease contributes to UN SDG 2 — Zero Hunger by reducing crop loss, improving food security, and supporting sustainable agriculture>"
}}

Rules:
1. Return ONLY the JSON object. No markdown fences, no preamble.
2. If the plant is healthy, still return the JSON but set diagnosis to "No disease detected — plant appears healthy." and provide general care tips.
3. Keep each treatment step and prevention tip concise (one sentence).
"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a plant pathology expert AI assistant. "
                        "Always respond with valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=MODEL,
            temperature=0.3,
            max_tokens=1024,
        )

        raw = chat_completion.choices[0].message.content.strip()

        # Strip markdown code fences if the model wraps them anyway
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]  # remove opening fence
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]

        diagnosis = json.loads(raw)
        return diagnosis

    except json.JSONDecodeError:
        return {
            "disease_name": predicted_class,
            "diagnosis": "LLM returned non-JSON response. Raw output: " + raw,
            "treatment_steps": [],
            "prevention_tips": [],
            "sdg2_impact": "Unable to generate SDG 2 impact statement.",
        }
    except Exception as e:
        return {
            "disease_name": predicted_class,
            "diagnosis": f"Groq API error: {str(e)}",
            "treatment_steps": [],
            "prevention_tips": [],
            "sdg2_impact": "Unable to generate SDG 2 impact statement.",
        }


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = get_diagnosis("Tomato___Late_blight", 0.96)
    print(json.dumps(result, indent=2))
