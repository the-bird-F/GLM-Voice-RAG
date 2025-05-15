import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import sys
import json
import torch
import argparse
from tqdm import tqdm 

from rag_module.e2e_rag import RAG, ASR_RAG, E2E_RAG
from rag_module.rag_tools import Recorder
from spoken_chatbot.model import GLM_Voice


def main(args):
    with open(args.data_path, 'r', encoding='utf-8') as f:
        content = f.read()
    input_path = args.input_path
    spoken_chatbot = GLM_Voice(args)
    if args.oracle:
        result_dir = f"answer_data/simple_{args.rag}_oracle"
        info_path = f"answer_data/simple_info_{args.rag}_asr.json"
        with open(args.data_path, 'r', encoding='utf-8') as f:
            query = f.read()
        rag = RAG(args)
    elif args.rag == 'e2e':
        result_dir = f"answer_data/simple_{args.rag}"
        info_path = f"answer_data/simple_info_{args.rag}_asr.json"
        rag = E2E_RAG(args)
    else:
        result_dir = f"answer_data/simple_{args.rag}_asr"
        info_path = f"answer_data/simple_info_{args.rag}_asr.json"
        rag = ASR_RAG(args)    
    
    recorder = Recorder()

    try:         
        ######### Indexing #########
        rag.build(-1, [content], search_type = "similarity", search_kwargs = {"k": 4}, reset = True)
        
        ######### prepare query #########
        recorder.respond_timer.start()
        question_audio = spoken_chatbot.process_audio(input_path)
        if not args.oracle:
            query = input_path # speech 
        
        ######### retrieval #########
        if args.oracle or args.rag == 'e2e':
            recorder.retrieval_timer.start()
            retrieval_context = rag.retrieve(query)                                
            recorder.retrieval_timer.stop()
        else:
            recorder.retrieval_timer.start()
            retrieval_context, asr_result = rag.retrieve(query)
            recorder.retrieval_timer.stop()
                                
        ######### generation #########
        prompt_input = f"""Answer the question based on the following context:
{retrieval_context}
Question: {question_audio}"""
        user_input = prompt_input.strip()
        system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. " 
        prompt = f"<|system|>\n{system_prompt}<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        
        spoken_chatbot.generate(prompt, args.temperature, output_file=f"{result_dir}")
        recorder.respond_timer.stop()
        recorder.record_info([content], "", retrieval_context)

            
    except Exception as e:
        print(f"Error processing {e}")
            
        
    recorder.save(info_path, extend=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./speech_data/simple/information.txt") # text information path
    parser.add_argument("--input-path", type=str, default="./speech_data/simple/question.wav") # speech question(query) path

    parser.add_argument("--rag", type=str, choices=["e2e", "multi", "bce", "openai"], default="multi")
    parser.add_argument("--oracle", type=bool, default=False) 
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_token", type=int, default=2000)
    parser.add_argument("--flow-path", type=str, default="./glm-4-voice-decoder")
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    parser.add_argument("--tokenizer-path", type=str, default="THUDM/glm-4-voice-tokenizer")
    parser.add_argument("--device", type=str, default="cuda") 
    
    args = parser.parse_args()
    args.device = torch.device(args.device)

    main(args)
    print(f"========== Finished ==========")

