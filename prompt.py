from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch

if __name__ == '__main__':
    model_name = "/mnt/disks/chfuab/Llama-2-7b-hf"
    

    model = LlamaForCausalLM.from_pretrained(model_name).to("cuda:0")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    # sys_prompt = "You are a helpful assistant. You are given a description about a creature and a rule to determine what species the creature is. Your task is to apply the rule to the description and determine what species the creature is. Give an answer about what species the creature is. The answer should be one sentence long. If you don't know or have not enough information about the answer, explain why."
    sys_prompt = "You are given a question. Your task is to answer the question. If you don't know the answer, explain why."
    
    # user_prompt = "Rule: If the creature has brain then it's species is human. Description: The creature has brain. What species is the creature? Answer:"
    user_prompt = "What is ChatGPT?"

    prompt = f'''<s>[INST] <<SYS>>
    {sys_prompt}
    <</SYS>>
    {user_prompt} [/INST] '''

    encoding = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    # encoding = tokenizer(user_prompt, return_tensors="pt").to("cuda:0")

    model = model.eval()
    with torch.no_grad():
        generate_ids = model.generate(encoding.input_ids, max_length=256)
        result = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

    print(result)