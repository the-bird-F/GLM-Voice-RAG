# E2E RAG for GLM-4-Voice: A Case Study 

This project presents an **end-to-end retrieval-augmented generation (E2E RAG)** speech dialogue system that bypasses the traditional speech-to-text (ASR) step, enabling direct retrieval and generation of relevant textual information from speech input.

We build upon the GLM-4-Voice model and incorporate SONAR, Meta's cross-modal embedding tool, to achieve cross-modal and low-latency retrieval-augmented generation (RAG) for speech-based interactions.

[中文版说明 👉 点击这里](./README_zh.md)

![Model Architecture](./resources/e2erag.svg)

## 📝 Paper
[Enhancing Speech-to-Speech Dialogue Modeling with End-to-End Retrieval-Augmented Generation](https://arxiv.org/abs/2505.00028)

## ✨ Overview of the Base Models
### GLM-4-Voice
GLM-4-Voice is an end-to-end speech large model developed by Zhipu AI with the following features:
- Supports both Chinese and English speech understanding and generation
- Enables real-time streaming speech dialogue
- Allows customization of speech emotion, intonation, speaking rate, and more based on user instructions

**However**, since GLM-4-Voice lacks external knowledge retrieval capabilities, its performance is limited in complex QA tasks such as HotpotQA.


GLM-4-Voice consists of three main components:
* GLM-4-Voice-Tokenizer: Adds vector quantization to the  [Whisper](https://github.com/openai/whisper) encoder and is supervised-trained on ASR data to convert continuous speech into discrete tokens. On average, only 12.5 discrete tokens are needed per second of audio.
* GLM-4-Voice-Decoder: A streaming-capable speech decoder based on the [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) Flow Matching architecture that reconstructs continuous audio from discrete tokens. It can start generating speech with as few as 10 tokens, significantly reducing E2E latency.
* GLM-4-Voice-9B: An extension of [GLM-4-9B](https://github.com/THUDM/GLM-4) with additional pretraining and alignment in the speech modality, allowing it to understand and generate discrete speech tokens.
 
### SONAR：Cross-Modal Speech-Text Embedding
[SONAR](https://github.com/facebookresearch/SONAR) is a multilingual, cross-modal speech-text embedding toolkit developed by Meta, with the following capabilities:
- Supports multilingual speech and text inputs
- Maps both speech and text segments into the same vector space using language-specific encoders
- Enables fine-grained alignment and retrieval between speech and text

We leverage SONAR's encoding capabilities to enable effective cross-modal retrieval in our E2E-RAG system.

### Qwen-Omni (TODO) 🧪
We also plan to integrate [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/) as part of future experiments on multimodal capabilities.


## 🛠️ Environment Setup
1. Clone the Repository and Create Environmen

    ```shell
    cd GLM-Voice-RAG
    pip install -e .[jupyter,linux]   # Linux 
    # or
    pip install -e .[jupyter,non_linux]  # Windows/macOS 
    ```
    Another choice:
    ```bash
    conda create -n glm-voice python==3.11
    conda activate glm-voice 
    pip install -r requirements.txt
    ```


2. Download relatied Checkpoints
    ```shell
    sudo apt install git-lfs
    git lfs install
    git clone https://huggingface.co/THUDM/glm-4-voice-decoder
    ```

## 📚 Dataset
### HotpotQA
```shell
git clone https://github.com/hotpotqa/hotpot.git
```

### RGB
```shell
git clone https://github.com/chen700564/RGB.git
```

### Ours speech dataset
```shell
git clone https://huggingface.co/datasets/the-bird-F/HotpotQA_RGBzh_speech
```

## 🚀 Quick Start
We provide different running programs for different datasets, where we can choose to run E2E RAG or ASR RAG:

```shell
# simple (Your data)
python examples/glm_voice_simple.py --rag e2e

# HotpotQA
python examples/glm_voice_hotpot.py --rag e2e

# RGB
python examples/glm_voice_rgb.py --rag e2e
```

Additionally, we provide a retrieval augmentation strategy generated in two rounds, which can be run with the following command:
```shell
python examples/double_glm_voice_hotpot.py
```

## 🙏 Acknowledgements
This project builds upon evaluation work conducted with GLM-4-Voice. The original codebase can be found at:
* [GLM-4-Voice-test](https://github.com/cwx-worst-one/GLM-4-Voice-test).

## 📄 License

+ The use of GLM-4 model weights must comply with the [model license](https://huggingface.co/THUDM/glm-4-voice-9b/blob/main/LICENSE).

+ The code in this repository is released under the [Apache 2.0](LICENSE) license.


If you find this project helpful, feel free to ⭐️ Star and 🔁 Fork it!