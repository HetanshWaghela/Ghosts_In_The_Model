"""
This script is to verify if the environment is correct configured
"""

import sys
print (f"Python version: {sys.version}")


try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"Torch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
except Exception as e:
    print("Torch import failed.")
    print(f"Error: {e}")
    print("Fix: install dependencies via: pip install -r requirements.txt")
    sys.exit(1)


try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("Transformers library found")
except Exception as e:
    print("Transformers import failed.")
    print(f"Error: {e}")
    print("Fix: install dependencies via: pip install -r requirements.txt")
    print("If you see a huggingface-hub version error, make sure:")
    print("  transformers<5 and huggingface-hub<1 are installed (see requirements.txt).")
    sys.exit(1)

try:
    from peft import LoraConfig,get_peft_model
    print("PEFT library found")
except Exception as e:
    print("PEFT import failed.")
    print(f"Error: {e}")
    print("Fix: install dependencies via: pip install -r requirements.txt")
    sys.exit(1)

try:
    import huggingface_hub
    print(f"Hugging Face Hub version: {huggingface_hub.__version__}")
except Exception:
    # Not fatal; transformers will complain if it's missing/incompatible.
    pass
print("\nLoading GPT-2 Small...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
print(f"Model loaded successfully!")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nTesting forward pass to see if GPT2 is working correctly...")
test_input= "The Eiffel Tower is located in"
inputs = tokenizer(test_input, return_tensors="pt")
outputs= model(**inputs,output_hidden_states=True)

print(f"Input: '{test_input}'")
print(f"Input token IDs: {inputs['input_ids'].tolist()}")

print(f"No.of hidden states:{len(outputs.hidden_states)}")
print(f"Hidden state shape at each layer:{outputs.hidden_states[0].shape}")

logits= outputs.logits[0,-1,:]
top_token= torch.argmax(logits).item()
top_prob= torch.softmax(logits,dim=0)[top_token].item()
predicted_word= tokenizer.decode([top_token])

print(f"\nPredicted next word: '{predicted_word}' with probability: {top_prob:.4f}")
print("\nAll tests passed successfully!")


#Now we move onto model exploration