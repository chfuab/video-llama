from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch

if __name__ == '__main__':
    model_name = "/mnt/disks/chfuab/Llama-2-7b-hf"
    

    model = LlamaForCausalLM.from_pretrained(model_name).to("cuda:0")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    sys_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    user_prompt = "What are the pros/cons of ChatGPT vs Open Source LLMs?"
    # context = "Alice has brain."

    prompt = f'''<s>[INST] <<SYS>>
    {sys_prompt}
    <</SYS>>
    {user_prompt} [/INST] '''

    # encoding = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    encoding = tokenizer(user_prompt, return_tensors="pt").to("cuda:0")

    model = model.eval()
    with torch.no_grad():
        generate_ids = model.generate(encoding.input_ids, max_length=256)
        result = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

    print(result)