# pip install requests

import requests
import time
import csv

MODEL = "nvidia/nemotron-3-nano-4b"
URL = "http://localhost:1234/v1/chat/completions"

prompts = [
    "Leg in maximaal 3 zinnen uit wat een lokale LLM is.",
    "Noem 3 voordelen van LM Studio.",
    "Schrijf een korte Python-functie die twee getallen optelt."
]

results = []

for prompt in prompts:
    start_time = time.time()

    response = requests.post(
        URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
    )

    end_time = time.time()
    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    print("\nPROMPT:", prompt)
    print("ANTWOORD:", answer)

    results.append({
        "model": MODEL,
        "prompt": prompt,
        "response_length": len(answer),
        "time_seconds": round(end_time - start_time, 2),
        "answer": answer
    })

with open("lmstudio_api_resultaten.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["model", "prompt", "response_length", "time_seconds", "answer"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nKlaar. Resultaten opgeslagen in lmstudio_api_resultaten.csv")