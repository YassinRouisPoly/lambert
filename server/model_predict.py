from transformers import CamembertTokenizer, CamembertForMaskedLM
import torch
from pydantic import BaseModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None

def is_loaded() -> bool:
    return model is not None

def load_model(model_name="base"):
    global model, tokenizer
    model_dir = "../models/" + model_name
    tokenizer = CamembertTokenizer.from_pretrained(model_dir)
    model = CamembertForMaskedLM.from_pretrained(model_dir).to(device)
    model.eval()

def predict_next_word(text, top_k=5):
    if not is_loaded():
        load_model()

    model.eval()
    masked = text + " <mask>"
    inputs = tokenizer(masked, return_tensors="pt").to(device)

    mask_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = logits[0, mask_index, :]
    top_ids = torch.topk(scores, top_k, dim=1).indices[0]

    return [tokenizer.decode(i).strip() for i in top_ids]

def predict_this_word(text, top_k=10):
    if not is_loaded():
        load_model()

    model.eval()
    text = text.split(" ")
    masked = " ".join(text[:-1]) + " <mask>"
    inputs = tokenizer(masked, return_tensors="pt").to(device)

    mask_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = logits[0, mask_index, :]
    top_ids = torch.topk(scores, top_k, dim=1).indices[0]

    res = []

    for i in [tokenizer.decode(i).strip() for i in top_ids]:
        if i.startswith(text[-1].lower()):
            res.append(i)
    return res

def predict(text):
    if not is_loaded():
        load_model()

    text_splitted = text.split(" ")
    first_sentence = " ".join(text_splitted[:-1])
    last_word = text_splitted[-1]
    next_words = predict_next_word(text)
    this_words = predict_this_word(text, top_k=20)

    res = [
        -1, # type de prédiction
        this_words,
        next_words,
        this_words, first_sentence + " " + this_words[0] if len(this_words) >= 1 else last_word, #complete this sentence
        first_sentence + " " + last_word + " " + next_words[0] if len(next_words) >= 1 else "", #complete next sentence
    ]

    if text.endswith(r"[.-:!?;\s]"):
        res[0] = 1
    elif last_word in this_words:
        res[0] = 1
    else:
        if len(this_words) >= 1:
            res[0] = 0
        else:
            res[0] = 2
    return res
