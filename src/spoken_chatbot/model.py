import torch, torchaudio
import numpy as np
import re
import os
import sys

sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")

import uuid
from hyperpyyaml import load_hyperpyyaml
from collections import defaultdict
from transformers import AutoTokenizer, WhisperFeatureExtractor, AutoModel, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf 

from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token


class Fake():
    def __init__(self, args):
        self.model = None
    def process_audio(self, audio_path):
        return "<|begin_of_audio|>" + "<|end_of_audio|>"
    def generate(self, prompt, temperature=0.2, output_file="example"):
        return "test"


def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
                                         fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)


class AudioDecoder:
    def __init__(self, config_path, flow_ckpt_path, hift_ckpt_path, device="cuda"):
        self.device = device

        with open(config_path, 'r') as f:
            self.scratch_configs = load_hyperpyyaml(f)

        # Load models
        self.flow = self.scratch_configs['flow']
        self.flow.load_state_dict(torch.load(flow_ckpt_path, map_location=self.device, weights_only=False))
        self.hift = self.scratch_configs['hift']
        self.hift.load_state_dict(torch.load(hift_ckpt_path, map_location=self.device, weights_only=False))

        # Move models to the appropriate device
        self.flow.to(self.device)
        self.hift.to(self.device)
        self.mel_overlap_dict = defaultdict(lambda: None)
        self.hift_cache_dict = defaultdict(lambda: None)
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 5
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 1
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)

    def token2wav(self, token, uuid, prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_feat=torch.zeros(1, 0, 80), embedding=torch.zeros(1, 192), finalize=False):
        tts_mel = self.flow.inference(token=token.to(self.device),
                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                      prompt_token=prompt_token.to(self.device),
                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(
                                          self.device),
                                      prompt_feat=prompt_feat.to(self.device),
                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                                          self.device),
                                      embedding=embedding.to(self.device))

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid] is not None:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)

        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # _tts_mel=tts_mel.contiguous()
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)

            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            # if self.hift_cache_dict[uuid] is not None:
            #     tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            tts_speech = tts_speech[:, :-self.source_cache_len]

        else:
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
            del self.hift_cache_dict[uuid]
            del self.mel_overlap_dict[uuid]
            # if uuid in self.hift_cache_dict.keys() and self.hift_cache_dict[uuid] is not None:
            #     tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech, tts_mel

    def offline_inference(self, token):
        this_uuid = str(uuid.uuid1())
        tts_speech, tts_mel = self.token2wav(token, uuid=this_uuid, finalize=True)
        return tts_speech.cpu()

    def stream_inference(self, token):
        token.to(self.device)
        this_uuid = str(uuid.uuid1())

        # Prepare other necessary input tensors
        llm_embedding = torch.zeros(1, 192).to(self.device)
        prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32).to(self.device)

        tts_speechs = []
        tts_mels = []

        block_size = self.flow.encoder.block_size
        prev_mel = None

        for idx in range(0, token.size(1), block_size):
            # if idx>block_size: break
            tts_token = token[:, idx:idx + block_size]

            print(tts_token.size())

            if prev_mel is not None:
                prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                flow_prompt_speech_token = token[:, :idx]

            if idx + block_size >= token.size(-1):
                is_finalize = True
            else:
                is_finalize = False

            tts_speech, tts_mel = self.token2wav(tts_token, uuid=this_uuid,
                                                 prompt_token=flow_prompt_speech_token.to(self.device),
                                                 prompt_feat=prompt_speech_feat.to(self.device), finalize=is_finalize)

            prev_mel = tts_mel
            prev_speech = tts_speech
            print(tts_mel.size())

            tts_speechs.append(tts_speech)
            tts_mels.append(tts_mel)

        # Convert Mel spectrogram to audio using HiFi-GAN
        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()

        return tts_speech.cpu()


class GLM_Voice(): 
    def __init__(self, args):
        self.glm_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.audio_decoder = AudioDecoder(
            config_path=os.path.join(args.flow_path, "config.yaml"),
            flow_ckpt_path=os.path.join(args.flow_path, 'flow.pt'),
            hift_ckpt_path=os.path.join(args.flow_path, 'hift.pt'),
            device=args.device
        ) 
        self.glm_model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).eval().to(args.device)
        self.whisper_model = WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(args.device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path) 
        self.max_new_token = args.max_new_token
        self.top_p = args.top_p
        self.device = args.device

    def process_audio(self, audio_path):
        audio_tokens = extract_speech_token(self.whisper_model, self.feature_extractor, [audio_path])[0]
        if len(audio_tokens) == 0:
            raise ValueError("No audio tokens extracted")
        audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
        return "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
    
    def save_results(self, text, audio, output_file):
        os.makedirs(output_file, exist_ok=True)
        text_path = os.path.join(output_file, "output.txt")
        audio_path = os.path.join(output_file, "output.wav")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        audio = audio.squeeze().cpu()
        torchaudio.save(audio_path, audio.unsqueeze(0), 22050, format="wav")
    
    def generate(self, prompt, temperature=0.2, output_file="./answer_data/example"):
        inputs = self.glm_tokenizer([prompt], return_tensors="pt").to(self.device)
        response = self.glm_model.generate(
            **inputs,
            max_new_tokens=self.max_new_token,
            temperature=temperature,
            top_p=self.top_p
        )
        response_list = response[0].tolist()

        text_tokens, audio_tokens = [], []
        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        end_of_question_tokens = self.glm_tokenizer.encode('streaming_transcription\n')[-5:]

        start_index = next(i for i in range(len(response_list) - len(end_of_question_tokens) + 1)
                        if response_list[i:i+len(end_of_question_tokens)] == end_of_question_tokens) + len(end_of_question_tokens)

        for token_id in response[0][start_index:]:
            if token_id >= audio_offset:
                audio_tokens.append(token_id - audio_offset)
            else:
                text_tokens.append(token_id)

        tts_token = torch.tensor(audio_tokens, device=self.device).unsqueeze(0)
        tts_speech, _ = self.audio_decoder.token2wav(tts_token, uuid=str(uuid.uuid4()), finalize=True)
        complete_text = self.glm_tokenizer.decode(text_tokens, spaces_between_special_tokens=False)
        complete_text = complete_text.split("<|user|>")[0].strip()
        
        self.save_results(complete_text, tts_speech.squeeze(), output_file)
        
        return complete_text
    
    

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class Whisper():
    def __init__(self, args):
        model_id = "openai/whisper-large-v3"
        asr_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=asr_torch_dtype, low_cpu_mem_usage=True, use_safetensors=True).to(args.device)
        self.asr_processor = AutoProcessor.from_pretrained(model_id)
        self.asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.asr_model,
            tokenizer=self.asr_processor.tokenizer,
            feature_extractor=self.asr_processor.feature_extractor,
            torch_dtype=asr_torch_dtype,
            device=args.device,
        )
        
    def audio_to_text(self, file_path, lang = "en", target_sampling_rate=16000):
        audio_array, sampling_rate = sf.read(file_path, dtype='float32')
        if sampling_rate != target_sampling_rate:
            audio_array = torchaudio.functional.resample(
                torch.tensor(audio_array), sampling_rate, target_sampling_rate
            ).numpy()
        question_wav = {"array": audio_array, "sampling_rate": target_sampling_rate}
        forced_decoder_ids = self.asr_processor.get_decoder_prompt_ids(language=lang, task="transcribe")
        result = self.asr_pipe(question_wav, generate_kwargs={"language": lang, "forced_decoder_ids": forced_decoder_ids})
        return result["text"]


'''
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


class Qwen_Omni():
    def __init__(self, args):
        self.model = Qwen2_5OmniModel.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")
        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        self.device = args.device
        self.output = args.output_dir
        

    def generate(self, prompt, text, audios, images, videos):
        # set use audio in video
        USE_AUDIO_IN_VIDEO = True
        # Preparation for inference
        text = self.processor.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(prompt, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        inputs = self.processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # Inference: Generation of the output text and audio
        text_ids, audio = self.model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        sf.write(
            os.path.join(self.output,"output.wav"),
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        
        return text
'''

