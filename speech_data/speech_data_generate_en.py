import json
import os
from tqdm import tqdm
import azure.cognitiveservices.speech as speechsdk

# read HotpotQA dataset
data_path = "hotpot/hotpot_dev_distractor_v1.json"
with open(data_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# data files
# ==========================
# speech_data
# |_ which dataset
#   |_ speaker_id
#     |_ question_id
#       |_ question.wav
#       |_ answer.wav
#       |_ context_0.wav
#       |_ context_1.wav
#       |
#       ...
#       |_ context_9.wav
# ==========================

speech_key=os.environ.get("AZURE_SPEECH_KEY")
region=os.environ.get("AZURE_REGION")
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)

# English Female:en-US-JennyNeural 
speech_config.speech_synthesis_language='en-US'
speech_config.speech_synthesis_voice_name='en-US-JennyNeural' 
speech_path = 'speech_data/hotpot_dev_distractor_v1/en-US-JennyNeural'
os.mkdir(speech_path)

speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

with tqdm(dataset, desc=f"Epoch {len(dataset)}") as pbar:
    for entry in pbar:
        question_id = entry["_id"] 
        os.mkdir(f"{ speech_path }/{question_id}")
        
        speech_synthesis_result = speech_synthesizer.speak_text_async(entry["question"]).get()
        stream = speechsdk.AudioDataStream(speech_synthesis_result)
        stream.save_to_wav_file(f"{speech_path}/{question_id}/question.wav")
        
        speech_synthesis_result = speech_synthesizer.speak_text_async(entry["answer"]).get()
        stream = speechsdk.AudioDataStream(speech_synthesis_result)
        stream.save_to_wav_file(f"{speech_path}/{question_id}/answer.wav")
        
        # for idx in range(10):
        #     speech_synthesis_result = speech_synthesizer.speak_text_async(''.join(entry['context'][idx][1])).get()
        #     stream = speechsdk.AudioDataStream(speech_synthesis_result)
        #     stream.save_to_wav_file(f"{speech_path}/{question_id}/context_{idx}.wav")

print('Finish')