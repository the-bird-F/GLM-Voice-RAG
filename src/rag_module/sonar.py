import torch
import torchaudio
import torchaudio.transforms as T
from typing import Any, List

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline


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
