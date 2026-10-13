from groq import Groq

from configs.settings import load_settings

client = Groq(api_key=load_settings().groq.api_key.get_secret_value())
for model in sorted(client.models.list().data, key=lambda m: m.id):
    print(model.id)