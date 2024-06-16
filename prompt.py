from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch

if __name__ == '__main__':
    model_name = "/mnt/disks/chfuab/Llama-2-7b-hf"
    model = LlamaForCausalLM.from_pretrained(model_name).to("cuda:0")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    rule = "If the creature has brain, then it's species is human."
    description = "The creature has brain."
    rule2 = "If the creature has hair, it's species is mammal."
    description2 = "The creature has hair."
    rule3 = "If the creature has backbone, it's species is vertebrates."
    description3 = "The creature has backbone."


    sys_prompt = "You are a helpful assistant. You are given a description about a creature and a rule to determine what species the creature is. Your task is to apply the rule to the description and determine what species the creature is. Give an answer about what species the creature is. If you don't know or have not enough information about the answer, explain why."
    
    user_prompt = '''Determine what is the species of the creature according to below rule and description.
    Rule: {rule},
    Description: {description}'''.format(rule=rule, description=description)
    model_answer = "The creature's species is human."
    user_prompt_2 = '''Determine what is the species of the creature according to below rule and description.
    Rule: {rule2},
    Description: {description2}'''.format(rule2=rule2, description2=description2)
    model_answer_2 = "The creature's species is mammal"
    user_prompt_3 = '''Determine what is the species of the creature according to below rule and description.
    Rule: {rule3},
    Description: {description3}'''.format(rule3=rule3, description3=description3)    



    prompt = f'''<s>[INST] <<SYS>>
    {sys_prompt}
    <</SYS>>
    {user_prompt} [/INST] {model_answer}</s>
    <s>[INST] {user_prompt_2} [/INST]{model_answer_2}</s>
    <s>[INST] {user_prompt_3} [/INST]'''



    encoding = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    # encoding = tokenizer(user_prompt, return_tensors="pt").to("cuda:0")

    model = model.eval()
    with torch.no_grad():
        generate_ids = model.generate(encoding.input_ids, max_length=300, temperature=0.5)
        result = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

    print(result)