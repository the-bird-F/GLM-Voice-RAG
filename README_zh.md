<h1 align="center">
E2E RAG for GLM-4-Voice: A Case Study
</h1>

<p align="center">
本项目构建了具有 <strong>端到端检索增强生成（E2E RAG）能力</strong> 的语音对话系统，直接基于语音进行相关文本信息的检索与生成，绕过传统的“语音转文本”流程。
<br />
系统以 <a href="https://github.com/THUDM/GLM-4">GLM-4-Voice</a> 为基础，结合 Meta 提出的 <a href="https://github.com/facebookresearch/SONAR">SONAR</a> 跨模态嵌入器，实现语音-文本跨模态检索与生成。
</p>

<p align="center">
    <a href="https://github.com/your-org/GLM-Voice-RAG"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-项目主页-blue?logo=github"></a>
    <a href="https://arxiv.org/abs/2505.00028"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2505.00028-red?logo=arxiv"></a>
</p>

<p align="center">
    <a href="./README.md">👉 English Version</a>
</p>

<p align="center">
    <img src="./resources/e2erag.svg" alt="模型结构图" width="666"/>
</p>

---


## ✨ 模型组件介绍

### GLM-4-Voice

来自智谱 AI 的 GLM-4-Voice 是一款支持端到端语音输入与输出的大语言模型， 支持中英文语音理解与生成、流式语音对话、可控制语音情感、语速、语调等属性，但由于缺乏外部知识库检索机制，GLM-4-Voice 在如 HotpotQA 等复杂问答任务中存在性能瓶颈。

#### 架构组成：

- **GLM-4-Voice-Tokenizer**  
  基于 [Whisper](https://github.com/openai/whisper) Encoder + Vector Quantization，在有监督数据集上训练，将语音编码为平均每秒仅 12.5 个离散 token。

- **GLM-4-Voice-Decoder**  
  基于 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 结构的流式语音解码器，最少 10 个 token 即可开始解码，大幅降低响应延迟。

- **GLM-4-Voice-9B**  
  在 [GLM-4-9B](https://github.com/THUDM/GLM-4) 基础上进行多模态对齐，可理解并生成语音 token 序列。

---

###  SONAR：多模态语音-文本嵌入器

[SONAR](https://github.com/facebookresearch/SONAR) 是 Meta AI 提出的跨模态多语言嵌入工具，支持：

- 多语言语音与文本嵌入  
- 语音与文本共享同一嵌入空间  
- 精准语义对齐与检索  

在本项目中，SONAR 用于将语音输入映射至文本知识库空间，从而实现跨模态检索增强。

---
### 补充：其他语音-文本嵌入器

[CLAP](https://huggingface.co/laion/clap-htsat-unfused) 是 LAION 团队发布的用于音频与文本对齐的多模态对比学习模型，支持将音频和文本映射到同一语义空间。

---

### 可用的语音识别后端

我们的系统支持多种语音识别（ASR）工具，通过命令行参数或配置文件轻松切换，可根据需求灵活权衡**识别精度**、**运行效率**与**多语言**能力。

- Whisper ([openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3))
- Faster-Whisper ([faster-whisper](https://huggingface.co/guillaumekln/faster-whisper))
- MMS ([facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all))
- Wav2Vec2 ([facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h))
---


### 🧪 Qwen-Omni（后续计划）

我们计划后续支持 [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) 模型，扩展本系统在图像、视频等更多模态下的对话能力。

---

## 🛠️ 环境配置
1. 克隆项目与环境创建

    ```shell
    cd GLM-Voice-RAG
    pip install -e .[jupyter,linux]   # Linux 
    # or
    pip install -e .[jupyter,non_linux]  # Windows/macOS 
    ```

    另一个创建环境的选择：
    ```shell
    cd GLM-Voice-RAG
    conda create -n glm-voice python=3.11 
    conda activate glm-voice 
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
    ```

    注：请确保安装了 sonar-space==0.3.0rc1 版本，参考官方安装文档[here](https://github.com/facebookresearch/large_concept_model?tab=readme-ov-file#installing)
    ```shell
    sudo apt install libsndfile1 
    conda install -c conda-forge libsndfile
    ```


2. 需要手动下载相关模型 Checkpoints
    ```shell
    sudo apt install git-lfs
    git lfs install
    git clone https://huggingface.co/THUDM/glm-4-voice-decoder

    # 镜像
    # export HF_ENDPOINT=https://hf-mirror.com/
    # git clone https://hf-mirror.com/THUDM/glm-4-voice-decoder
    ```


## 📚 数据集
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


## 🚀 快速开始
针对不同的数据集，我们提供了不同的运行脚本，在此可以选择运行 E2E RAG 或 ASR RAG：

```shell
# simple (Your data)
python examples/glm_voice_simple.py --rag e2e

# HotpotQA
python examples/glm_voice_hotpot.py --rag e2e

# RGB
python examples/glm_voice_rgb.py --rag e2e
```

另外，我们还提供了双轮生成的检索增强策略，可运行脚本：
```shell
python examples/double_glm_voice_hotpot.py
```


## 🙏 致谢
本项目是基于GLM-4-Voice的测评工作进一步研究，原代码库是：
* [GLM-4-Voice-test](https://github.com/cwx-worst-one/GLM-4-Voice-test)

## 📄 协议说明

+ GLM-4 模型的权重的使用则需要遵循 [模型协议](https://huggingface.co/THUDM/glm-4-voice-9b/blob/main/LICENSE)。

+ 本开源仓库的代码则遵循 [Apache 2.0](LICENSE) 协议。


如果你觉得本项目有帮助，欢迎 Star ⭐️ & Fork 🔁！