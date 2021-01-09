from transformers import GPT2TokenizerFast, GPT2Tokenizer


def get_tokenizer(tokenizer_type=None, from_pretrained=True, add_padding_token=True):
    if tokenizer_type is None or (tokenizer_type.lower() == "hf_gpt2tokenizerfast" and from_pretrained):
        tok = GPT2TokenizerFast.from_pretrained('gpt2')
        if add_padding_token:
            tok.add_special_tokens({'pad_token': '<|padding|>'})
        return tok
    elif tokenizer_type.lower() == "hf_gp2tokenizer" and from_pretrained:
        tok = GPT2Tokenizer.from_pretrained('gpt2')
        if add_padding_token:
            tok.add_special_tokens({'pad_token': '<|padding|>'})
        return tok
    else:
        raise NotImplementedError('TODO: add custom tokenizers')
