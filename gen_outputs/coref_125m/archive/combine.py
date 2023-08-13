import json

# Open the two input files
with open("/home/kit/stud/ukmwn/master_thesis/evaluation/gpt-neo/gen_outputs/coref_125m/factual_prompts_gen.jsonl", "r") as input1, open("/home/kit/stud/ukmwn/master_thesis/evaluation/FactualityPrompt/prompts/fever_factual_final.jsonl", "r") as input2:
  # Open the output file
  with open("./output.jsonl", "w") as output:
    # Iterate over the lines in the two input files
    for line1, line2 in zip(input1, input2):
      # Load the JSON objects from the lines
      obj1 = json.loads(line1)
      obj2 = json.loads(line2)

      obj1 = obj1["generation"]
      obj2 = obj2["prompt"]
      
      # Combine the objects into a single object
      combined = {"prompt": obj2, "text": obj1}
      
      # Write the combined object to the output file
      output.write(json.dumps(combined))
      output.write("\n")
