import os
from openai import OpenAI
from openai import OpenAIError
from dotenv import load_dotenv

load_dotenv()

# Mets ta clé API dans une variable d'environnement
#   sous Linux/Mac : export OPENAI_API_KEY="ta_cle"
#   sous Windows (PowerShell) : $env:OPENAI_API_KEY="ta_cle"

MODELS_TO_TEST = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
]

def test_models():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Aucune clé trouvée dans OPENAI_API_KEY")
        return

    client = OpenAI(api_key=api_key)

    for model in MODELS_TO_TEST:
        try:
            print(f"⏳ Test du modèle {model}...")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Bonjour"}],
                max_tokens=5,
            )
            print(f"✅ Accès OK à {model}")
        except OpenAIError as e:
            print(f"❌ Pas d'accès à {model} : {e}")

if __name__ == "__main__":
    test_models()
