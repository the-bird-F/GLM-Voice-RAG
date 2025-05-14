import json
import os
from tqdm import tqdm
import azure.cognitiveservices.speech as speechsdk

# read RGB zh dataset
data_path = "./RGB/data/zh.json"
dataset = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
            dataset.append(obj)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# data files
# ==========================
# speech_data
# |_ which dataset
#   |_ speaker_id
#     |_ question_id
#       |_ question.wav
#       |_ answer.wav
# ==========================

speech_key=os.environ.get("AZURE_SPEECH_KEY")
region=os.environ.get("AZURE_REGION")
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)

# zh-CN 
speech_config.speech_synthesis_language='zh-CN'
speech_config.speech_synthesis_voice_name='zh-CN-XiaoxiaoNeural' 
speech_path = '/data/pengchao.feng/speech_data/RGB_zh/zh-CN-XiaoxiaoNeural'
os.mkdir(speech_path)

speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

with tqdm(dataset, desc=f"Epoch {len(dataset)}") as pbar:
    for entry in pbar:
        question_id = entry["id"] 
        os.mkdir(f"{ speech_path }/{question_id}")
        
        speech_synthesis_result = speech_synthesizer.speak_text_async(entry["query"]).get()
        stream = speechsdk.AudioDataStream(speech_synthesis_result)
        stream.save_to_wav_file(f"{speech_path}/{question_id}/question.wav")
        
        # speech_synthesis_result = speech_synthesizer.speak_text_async("xxxxx").get()
        # stream = speechsdk.AudioDataStream(speech_synthesis_result)
        # stream.save_to_wav_file(f"{speech_path}/{question_id}/answer.wav")
        
        # for idx in range(10):
        #     speech_synthesis_result = speech_synthesizer.speak_text_async(''.join(entry['context'][idx][1])).get()
        #     stream = speechsdk.AudioDataStream(speech_synthesis_result)
        #     stream.save_to_wav_file(f"{speech_path}/{question_id}/context_{idx}.wav")

print('Finish')