import sys
import json
import re
import string
from collections import Counter
import pickle
import os
# ROUGE-L
from rouge_score import rouge_scorer


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def eval(prediction_file, gold_file, means = 'rougeL'):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)
        
    scorer = rouge_scorer.RougeScorer([means], use_stemmer=True)
    total_score_f1 = 0 
    total_score_recall = 0
    total_score_precision = 0
    
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        if cur_id not in prediction:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, prediction[cur_id], dp['answer'])
            
            scores  = scorer.score(dp['answer'], prediction[cur_id])  
            total_score_f1 += scores[means].fmeasure
            total_score_recall+= scores[means].recall
            total_score_precision+= scores[means].precision
            
    N = len(gold)
    print(N)
    for k in metrics.keys():
        metrics[k] /= N

    print(metrics)
    print(f"ROUGE-L F1 Score: {total_score_f1/N}")
    print(f"ROUGE-L Recall: {total_score_recall/N}")
    print(f"ROUGE-L Precision: {total_score_precision/N}")
    
def evalf(prediction_path, gold_file, means = 'rougeL'):
    files = os.listdir(prediction_path)
    with open(gold_file) as f:
        gold = json.load(f)

    # means ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer([means], use_stemmer=True)
    total_score_f1 = 0 
    total_score_recall = 0
    total_score_precision = 0
    
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,}
    N = 0
    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        if cur_id not in files:
            # print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            with open(f'{prediction_path}/{cur_id}/output.txt', 'r', encoding='utf-8') as file:
                outputs = file.read()
            em, prec, recall = update_answer(
                metrics, outputs, dp['answer'])
            scores  = scorer.score(dp['answer'], outputs)  
            total_score_f1 += scores[means].fmeasure
            total_score_recall+= scores[means].recall
            total_score_precision+= scores[means].precision
            N += 1
    for k in metrics.keys():
        metrics[k] /= N

    print(N)
    print(metrics)
    print(f"ROUGE-L F1 Score: {total_score_f1/N}")
    print(f"ROUGE-L Recall: {total_score_recall/N}")
    print(f"ROUGE-L Precision: {total_score_precision/N}")
    

def show_the_question(idx, prediction_path, query = None):

    with open("../hotpot/hotpot_dev_distractor_v1.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions_id = [entry["_id"] for entry in dataset]
    questions = [entry['question'] for entry in dataset]
    tmp_context = [entry['context'] for entry in dataset]
    answers = [entry['answer'] for entry in dataset]
    true_context = [entry['supporting_facts'] for entry in dataset]

    def find_context(lst, target):
        sub = []
        for sublist in lst:
            if sublist[0] == target[0]:  
                sub = sublist[1]  
                break
        if len(sub) <= target[1]:
            return ""    
        return sub[target[1]]  

    if query is not None:
        for i,q in enumerate(questions):
            if(query in q):
                idx = i    
    
    useful_info = [find_context(tmp_context[idx], target) for target in true_context[idx]]
    print("question: ", questions[idx])
    print("answer: ", answers[idx])
    print("useful info: ", useful_info)
    
    
    with open(f'{prediction_path}/{questions_id[idx]}/output.txt', 'r', encoding='utf-8') as file:
        output = file.read()
    print("prediction: ", output)
    
    f1, prec, recall = f1_score(output, answers[idx])
    print("f1: ", f1)
    print("prec: ", prec)
    print("recall: ", recall)
    print("exact match: ", exact_match_score(output, answers[idx]))


if __name__ == "__main__":
    # evalute the text output (in files)
    prediction_path =["hotpot_openai","hotpot_bce", "hotpot_multi","hotpot_sonar"]
    gold_file = "../hotpot/hotpot_dev_distractor_v1.json"
    evalf(prediction_path[0], gold_file)
    
    # show the details of the idx-th question/answer
    idx = 0
    show_the_question(idx, prediction_path[0])
    