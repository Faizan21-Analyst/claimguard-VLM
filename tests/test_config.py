
from groq import Groq, APIError
from configs.settings import load_settings

s = load_settings().groq
client = Groq(api_key=s.api_key.get_secret_value())

for name in (s.model, s.fallback_model):
    print(f"\nTesting: {name}")

    try:
        res = client.chat.completions.create(
            model=name,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with one word: ready"
                }
            ],
            temperature=0,
            max_tokens=50,
        )

        choice = res.choices[0]
        print("Content:", repr(choice.message.content))
        print("Finish reason:", choice.finish_reason)
        print("Usage:", res.usage)

    except APIError as e:
        print("Groq API error:", e)
