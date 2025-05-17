from datasets import load_dataset, DatasetDict

# Load dataset
dataset = load_dataset("pubmed_qa", "pqa_labeled")

# Split
train_valid_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
dev_test = train_valid_test["test"].train_test_split(test_size=0.5, seed=42)

# Group into DatasetDict
dataset_splits = DatasetDict({
    "train": train_valid_test["train"],
    "dev": dev_test["train"],
    "test": dev_test["test"]
})

# Save to disk
dataset_splits.save_to_disk("./data")

# MetaCLI 데이터셋 생성
biosses = load_dataset("biosses")
glue_sst2 = load_dataset("glue", "sst2")
ag_news = load_dataset("ag_news")
trec = load_dataset("trec")

metacli = DatasetDict({
    "biosses": biosses["train"] if "train" in biosses else biosses,
    "glue_sst2": glue_sst2["train"],
    "ag_news": ag_news["train"],
    "trec": trec["train"]
})

# Save to disk
metacli.save_to_disk("./data/metacli")