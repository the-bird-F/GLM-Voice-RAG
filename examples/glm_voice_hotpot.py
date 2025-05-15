import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import sys
import json
import torch
import argparse
from tqdm import tqdm 
from langchain import hub

from rag_module.e2e_rag import RAG, ASR_RAG, E2E_RAG
from rag_module.rag_tools import Recorder, read_hotpot
from spoken_chatbot.model import GLM_Voice

def main(args, flag_idx = 0):
    questions_id, questions, tmp_context, true_context = read_hotpot(args.data_path)
    input_dir = args.input_path
    spoken_chatbot = GLM_Voice(args)
    if args.oracle:
        result_dir = f"answer_data/hotpot_{args.rag}_oracle"
        info_path = f"answer_data/data_info_{args.rag}_oracle.json"
        rag = RAG(args)
    elif args.rag == 'e2e':
        result_dir = f"answer_data/hotpot_{args.rag}"
        info_path = f"answer_data/data_info_{args.rag}.json"
        rag = E2E_RAG(args)
    else:
        result_dir = f"answer_data/hotpot_{args.rag}_asr"
        info_path = f"answer_data/data_info_{args.rag}_asr.json"
        rag = ASR_RAG(args)    
    
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
                    documents, metadatas = [], []
                    for sen in tmp_context[idx]:
                        documents.append(" ".join(sen[1]))  
                        metadatas.append(sen[0])
                    rag.build(idx, documents, metadatas, search_type = "similarity", search_kwargs = {"k": 4}, reset = True)
                    
                    ######### prepare query #########
                    recorder.respond_timer.start()
                    input_path = os.path.join(input_dir, qid, 'question.wav')
                    question_audio = spoken_chatbot.process_audio(input_path)
                    if args.oracle:
                        query = questions[idx]
                    else:
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
                    prompt_input = prompt_tool.invoke({'context': retrieval_context, 'question': question_audio}) 
                    user_input = prompt_input.to_string().strip()
                    system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. " 
                    prompt = f"<|system|>\n{system_prompt}<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
                    
                    spoken_chatbot.generate(prompt, args.temperature, output_file=f"{result_dir}/{qid}")
                    recorder.respond_timer.stop()
                    if (not args.oracle) and args.rag != 'e2e':
                        recorder.record_wer(questions[idx], asr_result)   
                    recorder.record_info(tmp_context[idx], true_context[idx], retrieval_context)

                    torch.cuda.empty_cache()
                    flag = 0
            break
            
        except Exception as e:
            print(f"Error processing {idx}: {e}")
            flag_idx = idx
            flag += 1
        
    recorder.save(info_path, extend=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./hotpot/hotpot_dev_distractor_v1.json") # hotpotQA path
    parser.add_argument("--input-path", type=str, default="./speech_data/hotpot_dev_distractor_v1/en-US-JennyNeural") # speech_data path

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
    flag_idx = 0
    main(args, flag_idx)
    print(f"========== Finished ==========")

