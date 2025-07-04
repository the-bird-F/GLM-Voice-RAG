<h1 align="center">
E2E RAG for GLM-4-Voice: A Case Study
</h1>

<p align="center">
Implementation for the paper <em>"Enhancing Speech-to-Speech Dialogue Modeling with End-to-End Retrieval-Augmented Generation"</em>.
<br />
</p>

<div align="center">
    <a href="https://github.com/your-org/GLM-Voice-RAG/" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Code-blue?logo=github" alt="Github Code"></a>
    <a href="https://arxiv.org/abs/2505.00028" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2505.00028-red?logo=arxiv" alt="arXiv Paper"></a>
</div>

<p align="center">
    <a href="./README_zh.md">👉 中文版说明</a>
</p>

<p align="center">
  <img src="./resources/e2erag.svg" alt="Model Architecture" width="666"/>
</p>


An **end-to-end retrieval-augmented generation (E2E RAG)** speech dialogue system that enables direct **speech-to-text generation with retrieval**, bypassing traditional ASR pipelines. Built on top of **GLM-4-Voice** and **SONAR** for **cross-modal**, **low-latency** interaction.

---


## ✨ Overview of the Base Models

### GLM-4-Voice

Developed by [Zhipu AI](https://github.com/THUDM), GLM-4-Voice supports: Chinese & English understanding and generation, Real-time streaming dialogue, Customizable tone, emotion, and speech rate. However, GLM-4-Voice lacks knowledge retrieval, limiting its performance on complex QA tasks.

**Architecture Highlights**:

- **Tokenizer**: Whisper encoder + vector quantization  
- **Decoder**: [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) based streaming audio generator  
- **GLM-4-Voice-9B**: Speech-aware version of [GLM-4-9B](https://github.com/THUDM/GLM-4)

---

### SONAR: Cross-Modal Embedding

[SONAR](https://github.com/facebookresearch/SONAR) by Meta supports:

- Multilingual speech/text input  
- Speech-text joint embedding in the same space  
- Fine-grained retrieval and alignment  

Used in this project for cross-modal **retrieval-augmented generation (RAG)**.

---
### Additional: Other speech-to-text embeddingers 

[CLAP](https://huggingface.co/laion/clap-htsat-unfused) is a multimodal contrastive learning model for aligning audio and text by mapping them into a shared semantic space, by LAION.

---

### Supported ASR Backends 

Our system supports multiple ASR (Automatic Speech Recognition) backends, which can be switched via command-line arguments, to flexibly balance **accuracy**, **efficiency**, and **language coverage**.

- Whisper ([openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3))
- Faster-Whisper ([faster-whisper](https://huggingface.co/guillaumekln/faster-whisper))
- MMS ([facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all))
- Wav2Vec2 ([facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h))

---



### 🧪 Qwen-Omni (Planned)

We plan to explore [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) for future **multimodal experiments**.

---

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
### HotpotQA & Speech
```shell
git clone https://github.com/hotpotqa/hotpot.git
git clone https://huggingface.co/datasets/the-bird-F/HotpotQA_RGBzh_speech
```

### RGB & Speech
```shell
git clone https://github.com/chen700564/RGB.git
git clone https://huggingface.co/datasets/the-bird-F/HotpotQA_RGBzh_speech
```

### Spoken-SQuAD
```shell
git clone https://github.com/Chia-Hsuan-Lee/Spoken-SQuAD
```

### VoxPopuli-QA
```shell
git clone https://github.com/facebookresearch/voxpopuli
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