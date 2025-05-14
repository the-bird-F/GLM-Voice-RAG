import time
import json
import numpy as np
import sys
import os
import jiwer
from rouge_score import rouge_scorer

def get_total_size(obj, seen=None):
    """
    Recursively calculates the total size of an object and its nested objects.
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:  
        return 0

    seen.add(obj_id)
    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(get_total_size(v, seen) for v in obj.values())
        size += sum(get_total_size(k, seen) for k in obj.keys())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_total_size(i, seen) for i in obj)

    return size


class Timer:  
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.perf_counter()

    def stop(self):
        self.times.append(time.perf_counter() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)


class Recorder:
    """
    The Recorder to record information when running the RAG
    """
    def __init__(self):
        self.retrieval_timer = Timer()
        self.respond_timer = Timer()
        self.data_size = []
        self.retrieval_recall = []
        self.retrieval_f1 = []
        self.word_error = []

    def record_info(self, documents, true_context, retrieval_context):
        """
        record information
        :param documents: documents obj
        :param true_context: true context str
        :param retrieval_context: retrieval context str
        :return: None
        """
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores  = scorer.score(true_context, retrieval_context)
        self.retrieval_recall.append(scores['rougeL'].recall) 
        self.retrieval_f1.append(scores['rougeL'].fmeasure) 
        self.data_size.append(get_total_size(documents))

    def record_wer(self, query, asr_result, method='wer'):
        """
        record word error rate
        :param query: query str
        :param asr_result: asr result str
        :param method: wer or cer
        :return: None
        """
        if method == 'wer':
            self.word_error.append(jiwer.wer(query, asr_result))   
        if method == 'cer':
            self.word_error.append(jiwer.cer(query, asr_result))

    def save(self, file_path, extend=True):
        """
        save data to json file
        :param file_path: file path
        :param extend: extend or not
        :return: None
        """
        if extend and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}

        def merge_dicts(dict1, dict2):
            merged = {key: dict1.get(key, []) + dict2.get(key, []) for key in set(dict1) | set(dict2)}
            return merged

        data_time = merge_dicts(existing_data
        , {
            "re_time": self.retrieval_timer.times,
            "respond_time": self.respond_timer.times,
            "re_recall": self.retrieval_recall,
            "re_f1": self.retrieval_f1,
            "word_error": self.word_error,
            "data_size": self.data_size
        })

        with open(file_path, 'w') as f:
            json.dump(data_time, f)

           
def read_hotpot(data_path):
    """
    read hotpot dataset
    :param data_path: data path
    :return: questions_id, questions, tmp_context, true_context
    """
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    def find_context(lst, target):
        sub = []
        for sublist in lst:
            if sublist[0] == target[0]: 
                sub = sublist[1]  
                break
        if len(sub) <= target[1]:
            return ""    
        return sub[target[1]]  
    
    def merge_context(context, fact):
        true_context = ""
        for target in fact:
            true_context += find_context(context, target)
        return true_context
            
    questions_id = [entry["_id"] for entry in dataset]
    questions = [entry['question'] for entry in dataset]
    tmp_context = [entry['context'] for entry in dataset]
    true_context_idx = [entry['supporting_facts'] for entry in dataset]
    
    true_context = [merge_context(context, fact) for context, fact in zip(tmp_context, true_context_idx)]
    
    return questions_id, questions, tmp_context, true_context

