import random
random.seed(0)


def process_prompt(prefix, prompt, suffix):
    return prefix + prompt + suffix

def read_prompts(num_prompts=30):
    prompts = []
    with open("./PartiPrompts_Detail_eval/PartiPrompts_Detail.tsv", 'r') as fr:
        for line in fr:
            parts = line.strip().split("\t")
            assert len(parts) >= 3, parts
            prompts.append(parts[0])
        random.shuffle(prompts)

    if num_prompts == -1:
        return prompts
    return prompts[:num_prompts]
