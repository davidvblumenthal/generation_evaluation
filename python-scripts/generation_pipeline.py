from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

import datasets
import torch

from tqdm.auto import tqdm
from itertools import dropwhile
import json
import argparse, os

# write to file
def write_jsonl(file_name: str, generations_list: dict) -> None:
    
    with open(file_name, 'w') as file:
        for prompt, text in zip(generations_list["prompt"], generations_list["text"]):
            file.write(json.dumps({'prompt': prompt, 'text': text}) + '\n')

# Remove <|padding|> and <|endoftext|>
def delete_special_tokens(input_text: str, special_tokens: list):
    for special_token in special_tokens:
        input_text = input_text.replace(special_token, "")
    
    return input_text

# Decode Entity Tokenizer
def decode_entity_tokenizer(prompt_text, generated_token_ids, pad_token_id):
    # Encode the prompt
    prompt_token_ids = tokenizer.encode(prompt_text)
    
    # Delete the leading padding tokens from the batch generation
    generated_token_ids = list(dropwhile(lambda x: x == pad_token_id, generated_token_ids))
    # Return only addeded tokens
    generated_token_ids = generated_token_ids[len(prompt_token_ids):]

    # Decode the text
    decoded_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)

    # Delete <|padding|> and <|endoftext|> token from the decoded text
    decoded_text = delete_special_tokens(input_text=decoded_text, special_tokens=['<|padding|>', '<|endoftext|>'])

    return decoded_text




# Paths to the prompts
non_factual_path = "../../FactualityPrompt/prompts/fever_nonfactual_final.jsonl"
factual_path = "../../FactualityPrompt/prompts/fever_factual_final.jsonl"

# create datasets
def create_prompts_dataset(factual_path=factual_path, non_factual_path=non_factual_path, sample=False):
    # read jsonl files as HuggingFace datasets
    factual = datasets.load_dataset("json", data_files=factual_path, split="train")
    non_factual = datasets.load_dataset("json", data_files=non_factual_path, split="train")

    # for debugging purposes
    if sample:
        factual = factual.select(range(5))
        non_factual = non_factual.select(range(5))

    return factual, non_factual



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", help="Path to local model or Huggingface Model", type=str)
    parser.add_argument("--model_name", help="Internal name of the model for saving the output", type=str)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neo-125M")
    parser.add_argument("--output_path", type=str, help="Path to save output")

    parser.add_argument("--sampling", action=argparse.BooleanOptionalAction, 
                        help="if sampling then decode with p=0.9, else greedy decoding")
    
    parser.add_argument("--trained_with_padding",
                        action="store_true",
                        help="Model trained with dedicated paddig token or not"
                        )

    parser.add_argument("--return_full_text", action=argparse.BooleanOptionalAction, 
                        help="If set only added text is returned, otherwise full text is returned")
    parser.add_argument("--return_tensors", action=argparse.BooleanOptionalAction,
                        help="Set when Entity Tokenizer is used because decoding needs to handeld different")

    args = parser.parse_args()

    # construct the save path
    fact_save_dir = os.path.join(args.output_path, args.model_name)
    non_fact_save_dir = os.path.join(args.output_path, args.model_name)

    if args.sampling:
        fact_save_dir = os.path.join(args.output_path, args.model_name, 'sampling')
        non_fact_save_dir = os.path.join(args.output_path, args.model_name, 'sampling')
    
    fact_save_path = os.path.join(fact_save_dir, "factual-gen.jsonl")
    non_fact_save_path = os.path.join(non_fact_save_dir, "nonfactual-gen.jsonl")

    # create save_dir if not exists
    if not os.path.exists(fact_save_dir):
        os.makedirs(fact_save_dir)
        print(f"Repository created... Result files will be saved here: {fact_save_dir}")
    
    if not os.path.exists(non_fact_save_dir):
        os.makedirs(non_fact_save_dir)
        print(f"Repository created... Result files will be saved here: {non_fact_save_dir}")

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # set pad token for batch inference   
    if args.trained_with_padding:
        print(f"Using the dedicated padding token!")
        tokenizer.pad_token = tokenizer.pad_token
    else:
        print("Using EOS token as pad token!")
        tokenizer.pad_token = tokenizer.eos_token
    
    
    tokenizer.padding_side="left"

    print(f"Using padding side: {tokenizer.padding_side} and pad token: {tokenizer.pad_token}")
    print(f"Pad token has id: {tokenizer.pad_token_id}")

    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    # Construct generation pipeline
    # specify the generation arguments
    if args.sampling:
        print("Generation is using nucleus sampling with p=0.9!!!")
        generate_kwargs={
                        'do_sample': True,  # generate_kwargs do_sample=True, max_length=50, top_k=50
                        'top_p': 0.9,
                        'top_k': 0
                        }
    else:
        print("\nGeneration is using greedy decoding!!!")
        generate_kwargs={
                        'do_sample': False,  # generate_kwargs do_sample=True, max_length=50, top_k=50
                        'num_beams': 1,
                        'pad_token_id': tokenizer.pad_token_id
                        }
    # number of new tokens to generate
    max_new_tokens = 150
    
    # create pipeline
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device="cuda:0", #  cpu
                    max_new_tokens=max_new_tokens,
                    batch_size=62,
                    **generate_kwargs
                    ) 

    # Create Huggingface Dataset from prompt jsonl file
    # Factual prompts
    factual, non_factual = create_prompts_dataset(factual_path, non_factual_path, sample=False)

    #factual = factual.select(range(0,2)) # sample for debugging
    #non_factual = non_factual.select(range(0,2)) # sample for debugging

    # Set generation settings
    if args.return_full_text == None:
        print("Using generation strategy where only the generated text is returned!!!")
        args.return_full_text = False

    if args.return_tensors:
        args.return_full_text = None
        print(f"You are using the decoding from the Entity Tokenizer!! return_text is set to: {args.return_full_text}")
    # auskommentiert

    # Result list
    generations = {"prompt": [], "text": []}

    # KeyDataset (only *pt*) will simply return the item in the dict returned by the dataset item
    # as we're not interested in the *target* part of the dataset. For sentence pair use KeyPairDataset
    for out, prompt in tqdm(zip(pipe(KeyDataset(factual, "prompt"), return_tensors=args.return_tensors, return_full_text=args.return_full_text), factual)):
        #print(out)
        if args.return_tensors:
            # Construct the generated text
            generated_text = decode_entity_tokenizer(prompt_text=prompt["prompt"], generated_token_ids=out[0]["generated_token_ids"], pad_token_id=tokenizer.pad_token_id)
            #print(generated_text)
            # Add to result dict
            generations["text"].append(generated_text)       
            generations["prompt"].append(prompt["prompt"])

        else:
            generations["text"].append(out[0]["generated_text"])
            generations["prompt"].append(prompt["prompt"])
            # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
            # {"text": ....}

    # write to file
    write_jsonl(file_name=fact_save_path, generations_list=generations)
    
    # auskommentiert ende

    # Run through the non_factual
    generations = {"prompt": [], "text": []}

    for out, prompt in tqdm(zip(pipe(KeyDataset(non_factual, "prompt"), return_tensors=args.return_tensors, return_full_text=args.return_full_text), non_factual)):
        
        if args.return_tensors:
            # Construct the generated text
            generated_text = decode_entity_tokenizer(prompt_text=prompt["prompt"], generated_token_ids=out[0]["generated_token_ids"], pad_token_id=tokenizer.pad_token_id)
            # Add to result dict
            generations["text"].append(generated_text)       
            generations["prompt"].append(prompt["prompt"])
        
        else:
            generations["text"].append(out[0]["generated_text"])
            generations["prompt"].append(prompt["prompt"])
            # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
            # {"text": ....}

    # write to file
    write_jsonl(file_name=non_fact_save_path, generations_list=generations)





