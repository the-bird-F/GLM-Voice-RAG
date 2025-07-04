import torch
import torchaudio
import torchaudio.transforms as T
from typing import Any, List

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from transformers import ClapProcessor, ClapModel
import numpy as np

class SONAR_Wav_Embeddings(Embeddings, BaseModel):
    """SONAR embedding model."""
    
    client_speech: Any = None 
    client_text: Any = None
    language: str = None

    def __init__(self, lan = "en", device = None, **kwargs: Any):
        super().__init__(**kwargs)
        if device is None:
            device = torch.device("cuda:0")
        self.language = lan
        
        # TODO: auto choose encoder，tokenizer    
        if self.language == "en":
            self.client_speech = SpeechToEmbeddingModelPipeline(
                encoder="sonar_speech_encoder_eng",
                device=device,
            )
        elif self.language == "zh":
            self.client_speech = SpeechToEmbeddingModelPipeline(
                encoder="sonar_speech_encoder_cmn",
                device=device,
            )
        else:
            raise ValueError("Language comes soon, please contact the author to add support for this language")
        
        self.client_text = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
        )


    def embed_documents(self, texts) -> List[List[float]]:
        """
        Compute doc embeddings
        :param texts: The list of texts to embed.
        :returns: List of embeddings.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        lang = "eng_Latn" if self.language == "en" else "zho_Hans"
        with torch.inference_mode():
            embeddings = self.client_text.predict(texts, source_lang=lang)

        return embeddings.tolist()


    def embed_query(self, input) -> List[float]:
        """
        Compute query embeddings 
        :param input: (raw_inp, sr)
        :return: embedding
        """
        raw_inp, sr = input
        resample = T.Resample(orig_freq=sr, new_freq=16000)
        inp = resample(raw_inp)
        
        with torch.inference_mode():
            embedding = self.client_speech.predict([inp])
        return (embedding.tolist())[0]


class SONAR_Embeddings(Embeddings, BaseModel):
    """SONAR embedding model (without speech encoder)."""
    
    client_speech: Any = None 
    client_text: Any = None
    language: str = None

    def __init__(self, lan = "en", device = None, **kwargs: Any):
        super().__init__(**kwargs)
        if device is None:
            device = torch.device("cuda:0")
        self.language = lan
        
        self.client_text = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
        )


    def embed_documents(self, texts) -> List[List[float]]:
        """
        Compute doc embeddings
        :param texts: The list of texts to embed.
        :returns: List of embeddings.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        lang = "eng_Latn" if self.language == "en" else "zho_Hans"
        with torch.inference_mode():
            embeddings = self.client_text.predict(texts, source_lang=lang)

        return embeddings.tolist()


    def embed_query(self, input) -> List[float]:
        """
        Compute query embeddings 
        :param input: text
        :return: embedding
        """
        return self.embed_documents([input])[0]


class CLAP_Embeddings:
    """
    CLAP embedding model: 支持音频与文本嵌入提取。
    """

    client_speech: Any = None
    client_text: Any = None
    model: Any = None
    processor: Any = None

    def __init__(self, device=None, model_id="laion/clap-htsat-unfused", **kwargs):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id).to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本的嵌入向量。
        :param texts: 文本列表
        :return: 每条文本的 512-d 向量
        """
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            embeddings = self.model.get_text_features(**inputs)
        return embeddings.cpu().tolist()

    def embed_query(self, input: Any) -> List[float]:
        """
        从 torchaudio.load() 得到的 waveform 和 sr，生成 CLAP 音频向量
        :param input: (waveform: torch.Tensor, sr: int)
        :return: 512-d 音频向量
        """
        waveform, sr = input  

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 48kHz and Reshape
        if sr != 48000:
            resampler = T.Resample(orig_freq=sr, new_freq=48000)
            waveform = resampler(waveform)
        waveform_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)

        min_samples = 48000  
        if waveform_np.shape[0] < min_samples:
            repeat = int(np.ceil(min_samples / waveform_np.shape[0]))
            waveform_np = np.tile(waveform_np, repeat)

        with torch.no_grad():
            inputs = self.processor(
                audios=waveform_np,
                sampling_rate=48000,
                return_tensors="pt"
            ).to(self.device)

            embedding = self.model.get_audio_features(**inputs)

        return embedding[0].cpu().tolist()