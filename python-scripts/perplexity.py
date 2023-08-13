import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

import statistics
import math
import json
import argparse
from tqdm import tqdm
import logging

# Set up logging
from datasets.utils.logging import set_verbosity_info
set_verbosity_info()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define a custom logging message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add a console handler to send logging messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def move_tensor_to_cpu(tensor):
    """
    Given a PyTorch tensor, moves it from CUDA to the CPU and casts it to a Python datatype.
    """
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy().tolist()
    else:
        return tensor.detach().numpy().tolist()


def save_list_to_json(dictionary, file_path):
    """
    Given a list of data and a file path, saves the list as a JSON file at the specified path.
    """
    with open(file_path, 'a') as f:
        json.dump(dictionary, f, indent=2)


def tokenize(sample):
    outputs = tokenizer(sample["text"])

    batch_input_ids = []
    batch_attention_masks = []

    for input_id, attention_mask in zip(outputs["input_ids"], outputs["attention_mask"]):
        batch_input_ids.append(input_id)
        batch_attention_masks.append(attention_mask)

    return {"input_ids": batch_input_ids, "attention_mask": batch_attention_masks}


def describe_data(data):
    data = [num for num in data if not math.isnan(num)]
    mean = statistics.mean(data)
    stdev = statistics.stdev(data)
    highest = max(data)
    lowest = min(data)
    return mean, stdev, highest, lowest

"""
dataset_mapping = {
    "perplexity_factualityprompts": load_dataset("davidvblumenthal/perplexity_factualityprompts"),
    "wikitext-103": load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
}
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", help="Path to local model or Huggingface Model", type=str)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neo-125M")
    parser.add_argument("--dataset", type=str, help="On which dataset to calculate the perplixity.")
    parser.add_argument("--output_path", type=str, help="Path to save output")

    args = parser.parse_args()

    device = "cuda"
    model_id = args.model_path
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id) 

    # load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    #test = dataset_mapping[args.dataset]()
    test = load_dataset("davidvblumenthal/perplexity_factualityprompts", split="train")
    logger.info(f"Finished loading dataset with the following stats: {test}")
    #test = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    #test = test.select(range(100000))
    logger.info("Starting to tokenize dataset...")
    
    #encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    
    
    encodings = test.map(
        tokenize, 
        batched=True, 
        batch_size=100, 
        num_proc=2,
        remove_columns=test.column_names
        )

    
    logger.info(f"Finished tokenization - input to model: {encodings}")

    # set torch format
    encodings.set_format(type="torch", columns=encodings.column_names)

    # Setting sliding window parameters
    max_length = model.config.max_position_embeddings
    stride = 2048

    logger.info(f"Using model max lenght of: {max_length} and a stride of: {stride}")

    ppl_overall = []
    
    for sample in tqdm(encodings):
    
        #seq_len = sample.input_ids.size(1)
        seq_len = len(sample["input_ids"])

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            
            input_ids = torch.unsqueeze(sample["input_ids"], 0)           
            input_ids = input_ids[:, begin_loc:end_loc].to(device)   #sample["input_ids"][begin_loc:end_loc].to(device)      #
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100   #[:-trg_len] = -100        #

            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        ppl_cpu = move_tensor_to_cpu(ppl)
        # append to overall results
        ppl_overall.append(ppl_cpu)

    #avg_ppl = statistics.mean(ppl_overall)
    mean, stdev, highest, lowest = describe_data(ppl_overall)
    
    logger.info(f"Calculated Mean Perplexity: {mean}, Stdv: {stdev}, highest: {highest}, lowest: {lowest}")

    save_list_to_json({"dataset": args.dataset, 
                       "avg_perplexity": mean, 
                       "stdv": stdev, "min": lowest, 
                       "max": highest, 
                       "doc_ppl": ppl_overall
                       }, 
                       args.output_path)



