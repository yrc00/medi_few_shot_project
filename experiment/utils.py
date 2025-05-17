# utils.py
import torch

def format_metaicl_prompt(task_name, query_example, k_examples, tokenizer, max_length=1024):
    """
    MetaICL 스타일로 prompt를 구성하고, input_ids / attention_mask / labels 를 반환합니다.
    labels는 정답 부분만 포함되며, prompt 부분은 -100으로 마스킹되어 loss 계산에서 제외됩니다.
    """

    def example_to_string(example, task_name, include_output=True):
        """example(dict)를 string으로 변환"""
        task_map = {
            "MedNLI": ("premise", "hypothesis", "label"),
            "PubMedQA": ("context", "question", "final_decision"),
        }
        if task_name in task_map:
            fields = task_map[task_name]
            input_str = f"{fields[0].capitalize()}: {example.get(fields[0], '')}\n"
            input_str += f"{fields[1].capitalize()}: {example.get(fields[1], '')}\nAnswer:"
            output_str = f" {example.get(fields[2], '')}"
        else:
            input_str = f"Question: {example.get('question', '')}\nAnswer:"
            output_str = f" {example.get('answer', '')}"

        return input_str + output_str if include_output else input_str

    # Support examples (k-shot) + Query (문제)로 구성된 prompt
    prompt_parts = [example_to_string(e, task_name) for e in k_examples]
    query_prompt = example_to_string(query_example, task_name, include_output=False)
    query_full = example_to_string(query_example, task_name, include_output=True)

    prompt_parts.append(query_prompt)
    prompt_str = "\n".join(prompt_parts)
    full_text = prompt_str + query_full[len(query_prompt):]

    # 토크나이즈: prompt와 전체 input 모두 토크나이즈
    tokenized_prompt = tokenizer(prompt_str, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    tokenized_full = tokenizer(full_text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")

    input_ids = tokenized_prompt["input_ids"]
    attention_mask = tokenized_prompt["attention_mask"]
    labels = tokenized_full["input_ids"]

    # prompt 부분과 padding 부분을 -100으로 마스킹
    labels_masked = labels.clone()
    labels_masked[labels_masked == tokenizer.pad_token_id] = -100
    labels_masked[:, :input_ids.shape[1]] = -100

    return input_ids.squeeze(0), attention_mask.squeeze(0), labels_masked.squeeze(0)

