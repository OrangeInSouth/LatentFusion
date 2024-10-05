
def find_common_elements(vocab_list):
    for i, v in enumerate(vocab_list):
        vocab_list[i] = set(v)
    common_tokens = vocab_list[0]
    for i in range(1, len(vocab_list)):
        common_tokens = common_tokens & vocab_list[i]
    common_tokens = list(common_tokens)
    return common_tokens


def get_common_vocab_list_by_tokenizer(tokenizer_list):
    vocab_dict_list = [tokenizer.get_vocab() for tokenizer in tokenizer_list]
    vocab_keys_list = [vd.keys() for vd in vocab_dict_list]
    
    common_tokens = find_common_elements(vocab_keys_list)
    common_vocab_list = []
    for token in common_tokens:
        common_vocab_list.append({
            "token": token,
            "index": [vocab_dict[token] for vocab_dict in vocab_dict_list]
        })
    return common_vocab_list
