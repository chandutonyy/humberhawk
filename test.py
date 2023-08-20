from googletrans import Translator
translator = Translator()
result = translator.translate("Welcome to our tutorial!", dest="fr")
print(f"Translating the following text:\n{result.origin}\nDetected language code is {result.src}")
print(f"Here's the result:\n{result.text}\nTarget language code is {result.dest}")   