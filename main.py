import torch
import time
import sounddevice as sd
from deep_translator import GoogleTranslator
from transformers import AutoModelForCausalLM, AutoTokenizer
from vosk import Model, KaldiRecognizer
import pyaudio

# Загрузка модели ИИ
model = AutoModelForCausalLM.from_pretrained(r"C:\Users\HP\Desktop\startup\model")
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\HP\Desktop\startup\model")
chat_history_ids = torch.tensor([]).long()

# Загрузка модели Silero TTS
language = 'ru'
model_id = 'v3_1_ru'
sample_rate = 48000
speaker = 'xenia'
put_yo = True
put_accent = True
device = torch.device('cpu')

tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language=language,
                              speaker=model_id)
tts_model.to(device)

# Загрузка модели Vosk для распознавания речи
vosk_model = Model(r"C:\Users\HP\Desktop\startup\vosk-model-ru")  # Замените на путь к вашей модели
recognizer = KaldiRecognizer(vosk_model, sample_rate)

# Настройка PyAudio для захвата звука
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=int(sample_rate), input=True, frames_per_buffer=4000)
stream.start_stream()


# Функция для синтеза и воспроизведения речи
def synthesize_and_play_speech(text):
    audio = tts_model.apply_tts(text=text,
                                speaker=speaker,
                                sample_rate=sample_rate,
                                put_accent=put_accent,
                                put_yo=put_yo)
    sd.play(audio, sample_rate)
    time.sleep(len(audio) / sample_rate)
    sd.stop()


# Функция для распознавания речи
def recognize_speech():
    print("Говорите что-нибудь...")
    while True:
        data = stream.read(4000)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            recognized_text = result.split('"text" : "')[-1].rstrip('"\n }')
            if recognized_text:
                return recognized_text


# Основной цикл
while True:
    user_input = recognize_speech()

    # Перевод с русского на английский
    translated_input = GoogleTranslator(source='ru', target='en').translate(user_input)

    # Генерация ответа
    new_user_input_ids = tokenizer.encode(translated_input + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids.shape[0] > 0:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
        repetition_penalty=1.2
    )

    # Перевод ответа на русский
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    translated_response = GoogleTranslator(source='en', target='ru').translate(response)

    print(f"DialoGPT: {translated_response}")

    # Синтез и воспроизведение речи
    synthesize_and_play_speech(translated_response)