import json

def compare_keys(json1, json2):
    keys1 = set(json1.keys())
    keys2 = set(json2.keys())
    print("Key 1 length", len(keys1))
    print("Key 2 length", len(keys2))
    return keys1 == keys2

def compare_keys_values(json1, json2):
    res = True
    for key in json1:
        if json1[key] != json2[key]:
            print(key, json1[key], json2[key])
            res = False
        print(key, json1[key], json2[key])
    return res

if __name__ == '__main__':

    with open('TheLastBen_Papercut_SDXL.json', 'r') as f:
        paper_cut = json.load(f)

    with open('TheLastBen_William_Eggleston_Style_SDXL.json', 'r') as f:
        william = json.load(f)

    with open('TheLastBen_Filmic.json', 'r') as f:
        filmic = json.load(f)

    # print(compare_keys(paper_cut, william))
    # print(compare_keys(paper_cut, {k:v for k,v in filmic.items() if "unet" in k }))

    with open('TheLastBen_Papercut_SDXL_key_match.json', 'r') as f:
        paper_cut_match = json.load(f)

    with open('TheLastBen_Papercut_SDXL_key_match_backup.json', 'r') as f:
        paper_cut_match_gold = json.load(f)
    
    print(compare_keys_values(paper_cut_match_gold, paper_cut_match))