import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import sys
import random, math
import numpy as np
import yaml
import json, tqdm, requests
import torch
import torchaudio
import argparse
import uuid
from transformers import AutoTokenizer, WhisperFeatureExtractor, AutoModel
import shutil 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 

from rag_module.sonar import SONAR_Wav_Embeddings
from rag_module.rag_tools import Recorder
from rag_module.e2e_rag import RAG, ASR_RAG, E2E_RAG
from spoken_chatbot.model import GLM_Voice, Fake, Whisper

def retrival_imformation(con_label, recontent):
    from rouge_score import rouge_scorer  
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores  = scorer.score(con_label, recontent)
    return scores['rougeL'].recall,  scores['rougeL'].fmeasure


def processdata(instance, noise_rate, passage_num, filename, correct_rate = 0):
    query = instance['query']
    ans = instance['answer']

    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num

    if '_int' in filename:
        for i in instance['positive']:
            random.shuffle(i)
        print(len(instance['positive']))
        docs = [i[0] for i in instance['positive']]
        if len(docs) < pos_num:
            maxnum = max([len(i) for i in instance['positive']])
            for i in range(1,maxnum):
                for j in instance['positive']:
                    if len(j) > i:
                        docs.append(j[i])
                        if len(docs) == pos_num:
                            break
                if len(docs) == pos_num:
                    break
        neg_num = passage_num - len(docs)
        if neg_num > 0:
            negative = instance['negative'][:neg_num]
            docs += negative
    elif '_fact' in filename:
        correct_num = math.ceil(passage_num * correct_rate)
        pos_num = passage_num - neg_num - correct_num
        indexs = list(range(len(instance['positive'])))
        selected = random.sample(indexs,min(len(indexs),pos_num))
        docs = [instance['positive_wrong'][i] for i in selected]
        remain = [i for i in indexs if i not in selected]
        if correct_num > 0 and len(remain) > 0:
            docs += [instance['positive'][i] for i in random.sample(remain,min(len(remain),correct_num))]
        if neg_num > 0:
            docs += instance['negative'][:neg_num]
    else:
        if noise_rate == 1:
            neg_num = passage_num
            pos_num = 0
        else:
            if neg_num > len(instance['negative']):
                neg_num = len(instance['negative'])
                pos_num = passage_num - neg_num
            elif pos_num > len(instance['positive']):
                pos_num = len(instance['positive'])
                neg_num = passage_num - pos_num
        

        positive = instance['positive'][:pos_num]
        negative = instance['negative'][:neg_num]

        docs = positive + negative

    random.shuffle(docs)
    
    return query, ans, docs

def checkanswer(prediction, ground_truth):
    prediction = prediction.lower()
    if type(ground_truth) is not list:
        ground_truth = [ground_truth]
    labels = []
    for instance in ground_truth:
        flag = True
        if type(instance)  == list:
            flag = False 
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = instance.lower()
            if instance not in prediction:
                flag = False
        labels.append(int(flag))
    return labels

def getevalue(results):
    results = np.array(results)
    results = np.max(results,axis = 0)
    if 0 in results:
        return False
    else:
        return True
            
def predict(input_path, speech_query, query, ground_truth, docs, model, system, instruction, temperature, dataset, rag, whisper = None, asr = False):

    '''
    label: 0 for positive, 1 for negative, -1 for not enough information

    '''  
    
    if len(docs) == 0:
        text = instruction.format(QUERY=speech_query, DOCS='')
        prompt = f"<|system|>\n{system}<|user|>\n{text}<|assistant|>streaming_transcription\n"
        prediction = model.generate(prompt, temperature)

    else:
        if rag == 'None':
            docs = '\n'.join(docs)
        else:
            if rag == 'sonar':
                query = torchaudio.load(input_path)
                embedding = SONAR_Wav_Embeddings(lan = 'zh')
            elif rag == 'multi':
                embedding = HuggingFaceEmbeddings(
                    model_name="intfloat/multilingual-e5-large",
                    model_kwargs={'device': 'cuda'},
                    encode_kwargs={'batch_size': 16, 'normalize_embeddings': False}
                )
            elif rag == 'bce':
                embedding = HuggingFaceEmbeddings(
                    model_name="maidalun1020/bce-embedding-base_v1",
                    model_kwargs={'device': 'cuda'},
                    encode_kwargs={'batch_size': 16, 'normalize_embeddings': False}
                )
            elif rag == 'openai':
                api_key =  os.environ.get("OPENAI_API_KEY")
                api_base = os.environ.get("OPENAI_API_BASE")
                embedding = OpenAIEmbeddings(openai_api_key=api_key, openai_api_base=api_base)
                
            # Retrieve model  
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n", "。", "！", "？", "；", "，"])
            persist_directory = f'./answer_data/RGB_db_{rag}/'
            docs_chunk = text_splitter.create_documents(docs)
            vectorstore = Chroma.from_documents(documents=docs_chunk, embedding=embedding,  persist_directory=os.path.join(persist_directory,query[:16]))
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
            
                
                
            recorder.retrieval_timer.start()
            if asr:
                asr_result = whisper.audio_to_text(input_path)
                contexts_input = retriever.invoke(asr_result) 
            else:   
                asr_result = '' 
                contexts_input = retriever.invoke(query)
            docs = "\n".join(doc.page_content for doc in contexts_input)
            recorder.retrieval_timer.stop()
            
            
        text = instruction.format(QUERY=speech_query, DOCS=docs)
                   
        prompt = f"<|system|>\n{system}<|user|>\n{text}<|assistant|>streaming_transcription\n"
        prediction = model.generate(prompt, temperature)
        recorder.respond_timer.stop()
            
    if 'zh' in dataset:
        prediction = prediction.replace(" ","")

    if '信息不足' in prediction or 'insufficient information' in prediction:
        labels = [-1]
    else:
        labels = checkanswer(prediction, ground_truth)
    
    factlabel = 0

    if '事实性错误' in prediction or 'factual errors' in prediction:
        factlabel = 1

    return labels, prediction, factlabel, docs, asr_result



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='zh', help='evaluetion dataset',choices=['en','zh','en_int','zh_int','en_fact','zh_fact'])
    parser.add_argument('--input-dir', type=str, default='speech_data/RGB_zh/zh-CN-XiaoxiaoNeural', help='speech input directory')
    
    parser.add_argument('--modelname', type=str, default='GLM-Voice',help='model name')
    parser.add_argument('--rag', type=str, default='bce',help='rag name',choices=['None','multi','bce','openai','sonar'])
    parser.add_argument('--asr', type=bool, default=False, help='use asr')
    
    parser.add_argument('--noise_rate', type=float, default=0.8)      ### [0, 0.2, 0.4, 0.6, 0.8]
    parser.add_argument('--correct_rate', type=float, default=0.0,help='rate of correct passages')
    parser.add_argument('--passage_num', type=int, default=10, help='number of external passages')
    parser.add_argument('--factchecking', type=bool, default=False, help='whether to fact checking')  
    
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_token", type=int, default=2000)
    parser.add_argument("--flow-path", type=str, default="./glm-4-voice-decoder")
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    parser.add_argument("--tokenizer-path", type=str, default="THUDM/glm-4-voice-tokenizer")
    parser.add_argument("--device", type=str, default="cuda") 

    args = parser.parse_args()
    
    temperature = args.temp
    modelname = args.modelname
    noise_rate = args.noise_rate
    passage_num = args.passage_num

    instances = []
    with open(f'RGB/data/{args.dataset}.json','r') as f:
        for line in f:
            instances.append(json.loads(line))
    if 'en' in args.dataset:
        resultpath = 'RGB/result-en'
    elif 'zh' in args.dataset:
        resultpath = 'RGB/result-zh'
    if not os.path.exists(resultpath):
        os.mkdir(resultpath)

    if args.factchecking:
        prompt = yaml.load(open('RGB/config/instruction_fact.yaml', 'r'), Loader=yaml.FullLoader)[args.dataset[:2]]
        resultpath = resultpath + '/fact'
    else:
        prompt = yaml.load(open('RGB/config/instruction.yaml', 'r'), Loader=yaml.FullLoader)[args.dataset[:2]]

    system ="User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. " + prompt['system']
    instruction = prompt['instruction']

    if 'GLM-Voice' in modelname:
        model = GLM_Voice(args)
    else:
        print("Without Model")
        model = Fake(args)

    if args.asr:
        asr_pipeline = Whisper(args)
    else:
        asr_pipeline = None
        
    filename = f'{resultpath}/prediction_{args.dataset}_{modelname}_noise{noise_rate}_passage{passage_num}_correct{args.correct_rate}_{args.rag}_{args.asr}.json'
    useddata = {}
    if os.path.exists(filename):
        with open(filename) as f:
            for line in f:
                data = json.loads(line)
                useddata[data['id']] = data
  
    results = []
    input_dir = args.input_dir
    
    recorder = Recorder()
    with open(filename,'w') as f:
        for instance in tqdm.tqdm(instances):
            if instance['id'] in useddata and instance['query'] == useddata[instance['id']]['query'] and instance['answer']  == useddata[instance['id']]['ans']:
                results.append(useddata[instance['id']])
                f.write(json.dumps(useddata[instance['id']], ensure_ascii=False)+'\n')
                continue
            try:
                random.seed(2333)
                if passage_num == 0:
                    query = instance['query']
                    ans = instance['answer']
                    docs = []
                else:
                    query, ans, docs = processdata(instance, noise_rate, passage_num, args.dataset, args.correct_rate)
                    query_label, ans_label, docs_label = processdata(instance, 0, 1, args.dataset, args.correct_rate)
                    docs_label = '\n'.join(docs_label)
                    
                # use speech query
                input_path =  os.path.join(input_dir, str(instance['id']), 'question.wav')
                
                recorder.respond_timer.start()
                speech_query = model.process_audio(input_path)
                
                label,prediction,factlabel,retrival_docs,asr_result = predict(input_path, speech_query, query, ans, docs, model,system, instruction,temperature,args.dataset,args.rag, asr_pipeline, asr = args.asr)

                if passage_num != 0:
                    recorder.record_info(docs, docs_label, retrival_docs)
                    if args.asr:
                        recorder.record_wer(query, asr_result, method='cer')
                                    
                instance['label'] = label
                newinstance = {
                    'id': instance['id'],
                    'query': query,
                    'ans': ans,
                    'label': label,
                    'prediction': prediction,
                    'docs': docs,
                    'noise_rate': noise_rate,
                    'factlabel': factlabel
                }
                results.append(newinstance)
                f.write(json.dumps(newinstance, ensure_ascii=False)+'\n')
            except Exception as e:
                print("Error:", e)
                continue
    

    recorder.save(f'{resultpath}/data_time_{args.dataset}_{modelname}_noise{noise_rate}_passage{passage_num}_correct{args.correct_rate}_{args.rag}_{args.asr}.json')     
    
    tt = 0
    for i in results:
        label = i['label']
        if noise_rate == 1 and label[0] == -1:
            tt += 1
        elif 0 not in label and 1 in label:
            tt += 1
    print(tt/len(results))
    scores = {
    'all_rate': (tt)/len(results),
    'noise_rate': noise_rate,
    'tt':tt,
    'nums': len(results),
    }
    if '_fact' in args.dataset:
        fact_tt = 0
        correct_tt = 0
        for i in results:
            if i['factlabel'] == 1:
                fact_tt += 1
                if 0 not in i['label']:
                    correct_tt += 1
        fact_check_rate = fact_tt/len(results)
        if fact_tt > 0:
            correct_rate = correct_tt/fact_tt
        else:
            correct_rate = 0
        scores['fact_check_rate'] = fact_check_rate
        scores['correct_rate'] = correct_rate
        scores['fact_tt'] = fact_tt
        scores['correct_tt'] = correct_tt

    

    json.dump(scores,open(f'{resultpath}/prediction_{args.dataset}_{modelname}_noise{noise_rate}_passage{passage_num}_correct{args.correct_rate}_{args.rag}_{args.asr}_result.json','w'),ensure_ascii=False,indent=4)
    shutil.rmtree(f'answer_data/RGB_db_{args.rag}/')