from gpt import GPTLanguageModel
import torch

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# with open('./dataset/processed_dataset.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

# here are all the unique characters that occur in this text

# chars = sorted(list(set(text)))

chars = ['\n', ' ', '!', '"', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '~', '\xa0', 'à', 'ä', 'é', 'ï', 'ü', 'ē', 'ū', 'ŭ', '–', '—', '‘', '’', '“', '”', '…']
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

model = GPTLanguageModel().to('cuda')
model.load_state_dict(torch.load('./haikus.pt'))
model.eval()
user_input = input("from what do you want to start ? ")

context = torch.tensor([encode(user_input)], dtype=torch.int32, device='cuda')
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))

dummy_input = torch.zeros(1, 64, dtype=torch.int32).to("cuda")
torch.onnx.export(model,
                dummy_input,
                "haikus.onnx",
                input_names=['inputs'],
                output_names=['outputs'], 
                verbose=False,
                opset_version=13,
                dynamic_axes={'inputs': {0: 'batch_size', 1: 'sequence_length'}, 'outputs': {0: 'batch_size', 1: 'sequence_length'}}
)