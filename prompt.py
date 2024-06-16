from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch

if __name__ == '__main__':
    model_name = "/mnt/disks/chfuab/Llama-2-7b-hf"
    

    model = LlamaForCausalLM.from_pretrained(model_name).to("cuda:0")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    rule = "If the creature has brain then it's species is human."
    description = "The creature has brain."

    sys_prompt = "You are a helpful assistant. You are given a description about a creature and a rule to determine what species the creature is. Your task is to apply the rule to the description and determine what species the creature is. Give an answer about what species the creature is. The answer should be one sentence long. If you don't know or have not enough information about the answer, explain why."
    
    user_prompt = '''Determine the species of the creature based on below rule and description.
    Rule: {rule} 
    Description: {description}'''.format(rule=rule, description=description)


    
    
    text = '''Do players understand how to play the game?
    A game’s first tutorial is a great place to start learning, but it’s important to see how well players understand the game’s mechanics. You can observe retention rates, how long players are playing, and if they return to the game after their first session. You can see if your tutorials are effectively teaching the player by tracking metrics such as “Did the player get to the end of tutorial?”, “Did they complete the first level?”, “What is the average number of sessions played during the soft launch period?”, and “What is the average progression rate?”
    How much can you spend for user acquisition?
    You can use the soft launch period to see how your user acquisition strategy works and how much player engagement costs. This will give you a better idea of how much you can spend to acquire new players when the game is released globally. Some metrics that you can use to measure user acquisition costs include: cost per install, cost per first session, cost per daily active user, and customer lifetime value.
    How much do players monetize?
    You may want to know how much players will spend on in-game items and currency to understand the game’s revenue potential. It’s important to keep in mind that revenue from a soft launch can be very different than a worldwide launch. You might want to look at metrics such as average revenue per user, average revenue per paying user, conversion rate, and revenue to cost ratio.
    How does the game stand out from the competition?
    It’s important to know how your game compares to other games in the same genre, especially games that have already been released. You can use the soft launch period to find out how your game stands out from the competition. This can help you identify areas where your game is lacking and areas where the game excels.
    How to Soft Launch
    Now that we’ve established what a soft launch is, why it’s important, and what you can learn from it, let’s talk about how to soft launch your game.
    Step 1: Choose Your Market
    The first step is choosing which market you want to soft launch in. You want to choose a market that is similar to your target audience, but with enough scale to give you meaningful metrics. You might want to consider a country with a large number of gamers, a country with a similar culture and demographics to your target audience, or a country where your game’s genre is popular.
    Step 2: Set Your Testing Goals
    Next, you want to set specific goals for what you want to test during the soft launch. These goals should be measurable and tied to specific metrics. For example, one goal could be to increase player retention by 10% during the soft launch period.
    '''


    p5 = """<s>[INST] <<SYS>>
    You are a researcher task in summarizing and writing concise brief of articles.  
    Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
    If you don't know the answer, please don't share false information.
    <</SYS>>
    Write a concise TL;DR summary in numeric bullet-points for the following article, 
    don't repeat ideas in bullet points. Limit the number of bullet-point to 5. article: {BODY}
    [/INST]""".format(BODY=text)






    prompt = f'''<s>[INST] <<SYS>>
    {sys_prompt}
    <</SYS>>
    {user_prompt} [/INST]'''



    encoding = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    # encoding = tokenizer(user_prompt, return_tensors="pt").to("cuda:0")

    model = model.eval()
    with torch.no_grad():
        generate_ids = model.generate(encoding.input_ids, max_length=1000, temperature=0.5)
        result = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

    print(result)