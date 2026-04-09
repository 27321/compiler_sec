import torch
from transformers import BertTokenizer, BertModel

torch.manual_seed(0)


bert_tokenizer = BertTokenizer.from_pretrained(
        "/workspace/Documents/compiler_wjk2/test/bert_base_uncased"
    ) 
bert_model = BertModel.from_pretrained(
        "/workspace/Documents/compiler_wjk2/test/bert_base_uncased"
    )
bert_model.eval()

text = "This is a fixed input for compiler differential testing."

encoded = bert_tokenizer(
    text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=16,
)

input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]
token_type_ids = encoded["token_type_ids"]

torch.save(
    {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    },
    "fixed_input.pt",
)

print("Saved fixed_input.pt")
print(input_ids.shape, attention_mask.shape, token_type_ids.shape)
