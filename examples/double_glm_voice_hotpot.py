import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import sys
import json
import torch
import argparse
from tqdm import tqdm 
from langchain import hub

from rag_module.e2e_rag import RAG, ASR_RAG, E2E_RAG
from rag_module.rag_tools import Recorder, read_hotpot
from spoken_chatbot.model import GLM_Voice


def double_main(args, flag_idx = 0):
    questions_id, questions, tmp_context, true_context = read_hotpot(args.data_path)
    input_dir = args.input_path
    spoken_chatbot = GLM_Voice(args)

    result_dir = f"answer_data/hotpot_{args.rag}_double"
    info_path = f"answer_data/data_info_{args.rag}_double.json"
    rag = RAG(args)
    
    prompt_tool = hub.pull("rlm/rag-prompt")
    recorder = Recorder()
    flag = 0
    while(flag_idx < len(questions_id) and flag < 10):
        try:
            with tqdm(enumerate(questions_id), desc=f"Epoch {len(questions_id)}") as pbar:
                for idx, qid in pbar:
                    if idx < flag_idx:
                        continue  
                    
                    ######### Indexing #########
                    rag.build(idx, tmp_context[idx], search_type = "similarity", search_kwargs = {"k": 4}, reset = True)
                    
                    ######### prepare query #########
                    recorder.respond_timer.start()
                    input_path = os.path.join(input_dir, qid, 'question.wav')
                    question_audio = spoken_chatbot.process_audio(input_path)
                    
                    prompt_input = f"""Think about what information is needed to answer the question "{question_audio}", and respond only that information needed in one sentence."""
                    user_input = prompt_input.strip()
                    system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. " 
                    prompt = f"<|system|>\n{system_prompt}<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
                    
                    query = spoken_chatbot.generate(prompt, args.temperature, output_file=f"{result_dir}/draft{qid}")
                    
                    ######### retrieval #########
                    recorder.retrieval_timer.start()
                    retrieval_context = rag.retrieve(query)                                
                    recorder.retrieval_timer.stop()
                                            
                    ######### generation #########
                    prompt_input = prompt_tool.invoke({'context': retrieval_context, 'question': question_audio}) 
                    user_input = prompt_input.to_string().strip()
                    system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. " 
                    prompt = f"<|system|>\n{system_prompt}<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
                    
                    spoken_chatbot.generate(prompt, args.temperature, output_file=f"{result_dir}/{qid}")
                    recorder.respond_timer.stop()

                    recorder.record_info(tmp_context[idx], true_context[idx], retrieval_context)

                    torch.cuda.empty_cache()
                    flag = 0
            break
            
        except Exception as e:
            print(f"Error processing {qid}: {e}")
            flag_idx = idx
            flag += 1
    recorder.save(info_path, extend=True)



def un_main(args, flag_idx = 0):
    questions_id, questions, tmp_context, true_context = read_hotpot(args.data_path)
    input_dir = args.input_path
    spoken_chatbot = GLM_Voice(args)
    
    e2e_rag = E2E_RAG(args)
    asr_rag = ASR_RAG(args)
    result_dir = f"answer_data/hotpot_{args.rag}_uncetainty"
    info_path = f"answer_data/data_info_{args.rag}_uncetainty.json"

    recorder = Recorder()
    flag = 0
    while(flag_idx < len(questions_id) and flag < 10):
        try:
            with tqdm(enumerate(questions_id), desc=f"Epoch {len(questions_id)}") as pbar:
                for idx, qid in pbar:
                    if idx < flag_idx:
                        continue  
                    
                    ######### Indexing #########
                    e2e_rag.build(idx, tmp_context[idx], search_type = "similarity", search_kwargs = {"k": 4}, reset = True)
                    asr_rag.build(idx, tmp_context[idx], search_type = "similarity", search_kwargs = {"k": 4}, reset = True)
                    
                    ######### prepare query #########
                    recorder.respond_timer.start()
                    input_path = os.path.join(input_dir, qid, 'question.wav')
                    question_audio = spoken_chatbot.process_audio(input_path)

                    query = input_path 
                    
                    ######### retrieval #########
                    recorder.retrieval_timer.start()
                    retrieval_context = e2e_rag.retrieve(query)                                
                    recorder.retrieval_timer.stop()
                                            
                    ######### generation #########
                    prompt_input = f"""Answer the question according to the content of the text. If the text information is insufficient, answer "insufficient information",
    Question:{question_audio}
    Context:{retrieval_context}"""
                    user_input = prompt_input.strip()
                    system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. " 
                    prompt = f"<|system|>\n{system_prompt}<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
                    
                    prediction = spoken_chatbot.generate(prompt, args.temperature, output_file=f"{result_dir}/{qid}")
                    asr_result = ""
                    if 'insufficient information' in prediction or 'not sure' in prediction:
                        ######### retrieval again #########
                        retrieval_context_plus, asr_result= asr_rag.retrieve(query) 
                        
                        ######### generation again #########  
                        prompt_input = f"""Answer the question according to the content of the text. If the text information is insufficient, answer "insufficient information",
    Question:{question_audio}
    Context:{retrieval_context + retrieval_context_plus}"""
                        user_input = prompt_input.strip()
                        prompt = f"<|system|>\n{system_prompt}<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

                        spoken_chatbot.generate(prompt, args.temperature, output_file=f"{result_dir}/{qid}")
                    
                    recorder.respond_timer.stop()

                    recorder.record_wer(questions[idx], asr_result)    
                    recorder.record_info(tmp_context[idx], true_context[idx], retrieval_context)

                    torch.cuda.empty_cache()
                    flag = 0
            break
            
        except Exception as e:
            print(f"Error processing {qid}: {e}")
            flag_idx = idx
            flag += 1
    recorder.save(info_path, extend=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./hotpot/hotpot_dev_distractor_v1.json") # hotpotQA path
    parser.add_argument("--input-path", type=str, default="./speech_data/hotpot_dev_distractor_v1/en-US-JennyNeural") # speech_data path

    parser.add_argument("--mode", type=str, choices=["double", "uncertainty"], default="uncertainty")
    parser.add_argument("--rag", type=str, choices=["multi", "bce", "openai"], default="multi")
    
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
    
    flag_idx = 0
    if args.mode == "uncertainty":
        un_main(args, flag_idx)
    else:
        double_main(args, flag_idx)
    print(f"========== Finished ==========")

