import random
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from sklearn.metrics import accuracy_score

# 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
k_values = [0, 1, 2, 4, 8, 16]
ia3_model_path = "../models/qwen_pubmedqa_ia3_sft/final"
metaicl_base_path = "../models/metaicl_qwen_meta"
huggingface_model_id = "Qwen/Qwen3-1.7B"
data_path = "../data"

# 데이터 로딩
dataset = load_from_disk(f"{data_path}")
test_set = dataset["test"]
train_set = dataset["dev"]

# 텍스트 생성 함수
def generate(model, tokenizer, prompt, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

# 평가 함수
def evaluate(model, tokenizer, support_examples, test_set, result_file):
    preds, labels, results = [], [], []

    for example in test_set:
        prompt = ""
        for support in support_examples:
            prompt += f"Q: {support['question']}\nC: {support['context']}\nA: {support['long_answer']}\n\n"
        prompt += f"Q: {example['question']}\nC: {example['context']}\nA:"
        answer = generate(model, tokenizer, prompt)
        pred = "yes" if "yes" in answer.lower() else ("no" if "no" in answer.lower() else "maybe")

        preds.append(pred)
        labels.append(example["long_answer"])
        results.append({
            "question": example["question"],
            "context": example["context"],
            "prediction": pred,
            "raw_answer": answer,
            "label": example["long_answer"]
        })

    acc = accuracy_score(labels, preds)

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return acc, preds, labels


# k-shot 지원 예제 미리 샘플링 (seed 고정)
random.seed(42)
train_list = list(train_set)

support_examples_dict = {}
for k in k_values:
    if k > 0:
        support_examples_dict[k] = random.sample(train_list, k)
    else:
        support_examples_dict[k] = []

# MetaICL 모델 평가
for k in k_values:
    print(f"\n==== Evaluating MetaICL Model (k={k}) ====")
    model_path = f"{metaicl_base_path}/k_{k}/final"
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    support_examples = support_examples_dict[k]
    result_file = f"../results/results_metaicl_k{k}.json"
    acc, preds, labels = evaluate(model, tokenizer, support_examples, test_set, result_file)
    print(f"[MetaICL] Accuracy (k={k}): {acc:.4f}")


# IA3 SFT 모델 평가
print("\n==== Evaluating IA3 SFT Model ====")
model = AutoModelForCausalLM.from_pretrained(ia3_model_path, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(ia3_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

result_file = "../results/results_ia3.json"
acc, preds, labels = evaluate(model, tokenizer, [], test_set, result_file)
print(f"[IA3 SFT] Accuracy: {acc:.4f}")


# Original Hugging Face Qwen 모델 평가
print("\n==== Evaluating Original Hugging Face Qwen Model ====")
model = AutoModelForCausalLM.from_pretrained(huggingface_model_id, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(huggingface_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

result_file = "../results/results_qwen_hf.json"
acc, preds, labels = evaluate(model, tokenizer, [], test_set, result_file)
print(f"[Original Qwen (HF)] Accuracy: {acc:.4f}")
