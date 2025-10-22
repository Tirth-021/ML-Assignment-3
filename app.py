import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from pathlib import Path

# --- Configuration ---
# This MUST match the folder where your models are saved.
ARTIFACTS_DIR = Path(".")
DEFAULT_CONTEXT = 10
DEFAULT_EMBED_DIM = 128
DEVICE = "cpu" # Force CPU for compatibility

# ----------------------------------------------------------------------
# 1. EXACT Model Definition (Copied from your training notebook)
# ----------------------------------------------------------------------
# This class MUST be identical to the one you used for training.
# Note: Your assignment asks for 1-2 hidden layers[cite: 32]. 
# This code assumes the 2-layer model from your notebook.

class NextWordMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, context, h1, h2, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * context, h1),
            nn.LayerNorm(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, vocab_size)
        )
    def forward(self, x):
        e = self.embed(x).view(x.size(0), -1)
        return self.net(e)

# ----------------------------------------------------------------------
# 2. Generation Helper Functions
# ----------------------------------------------------------------------

def sample_top_k(probs, k):
    """Samples an index from the top-k probabilities."""
    topk_idx = np.argsort(probs)[-k:]
    topk_probs = probs[topk_idx]
    topk_probs = topk_probs / (topk_probs.sum() + 1e-12) # Re-normalize
    return np.random.choice(topk_idx, p=topk_probs)

@torch.no_grad()
def generate_sequence(model, stoi, itos, context_words, num_to_gen, temperature, top_k, model_params):
    """
    Autoregressively generates a sequence of 'k' words.
    """
    
    context_size = model_params['context']
    # Handle unknown words from user input [cite: 49]
    unk_idx = stoi.get('<unk>', 0) 
    pad_idx = stoi.get('<pad>', 1) 

    # 1. Tokenize and pad the input context
    idxs = [stoi.get(w, unk_idx) for w in context_words]
    
    generated_words = []

    for _ in range(num_to_gen):
        # 2. Prepare the tensor for the model
        if len(idxs) < context_size:
            context_tensor = [pad_idx] * (context_size - len(idxs)) + idxs
        else:
            context_tensor = idxs[-context_size:]
        
        xb = torch.tensor([context_tensor], dtype=torch.long).to(DEVICE)
        
        # 3. Get model output (logits)
        logits = model(xb)[0]
        
        # 4. Apply temperature control [cite: 48]
        logits = logits / max(1e-8, temperature)
        
        # 5. Get probabilities
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        
        # 6. Sample from Top-K
        next_idx = sample_top_k(probs, k=top_k)
        
        # 7. Add the new word to our context and output
        idxs.append(next_idx)
        generated_words.append(itos[next_idx])

    return " ".join(generated_words)

# ----------------------------------------------------------------------
# 3. Streamlit Caching and Model Loading
# ----------------------------------------------------------------------

@st.cache_resource
def load_model(model_path):
    """
    Loads the model and vocab. This is cached so it only runs once.
    """
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Recreate the exact model architecture
        vocab_size = len(checkpoint['itos'])
        
        # This resolves the assignment contradiction: we give options [cite: 51]
        # and display the params of the loaded variant[cite: 47].
        if 'cat1' in model_path.name:
            model_params = {
                'context': DEFAULT_CONTEXT,
                'embed_dim': DEFAULT_EMBED_DIM,
                'h1': 512,
                'h2': 256,
                'dropout': 0.3
            }
        elif 'cat2' in model_path.name:
            model_params = {
                'context': DEFAULT_CONTEXT,
                'embed_dim': DEFAULT_EMBED_DIM,
                'h1': 256,
                'h2': 128,
                'dropout': 0.3
            }
        
        model = NextWordMLP(
            vocab_size=vocab_size,
            embed_dim=model_params['embed_dim'],
            context=model_params['context'],
            h1=model_params['h1'],
            h2=model_params['h2'],
            dropout=model_params['dropout']
        )
        
        model.load_state_dict(checkpoint['model'])
        model.to(DEVICE)
        model.eval()
        
        return model, checkpoint['stoi'], checkpoint['itos'], model_params

    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Make sure 'q1_artifacts' folder exists and models are downloaded via Git LFS.")
        return None, None, None, None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None, None, None, None

# ----------------------------------------------------------------------
# 4. Streamlit UI
# ----------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("Next-Word Prediction with MLP [Inference-Only]")

# --- Sidebar Controls ---
st.sidebar.title("Controls")

# Let user choose which pre-trained model variant to use [cite: 51]
model_options = {
    "Category 1: Natural (Shakespeare)": ARTIFACTS_DIR / 'cat1_model.pt',
    "Category 2: Structured (Linux C Code)": ARTIFACTS_DIR / 'cat2_model.pt'
}
model_choice = st.sidebar.selectbox("Choose Model Variant", model_options.keys())
model_path = model_options[model_choice]

# Load the selected model
model, stoi, itos, params = load_model(model_path)

if model:
    # Display the loaded model's parameters [cite: 47]
    st.sidebar.subheader("Loaded Model Parameters")
    st.sidebar.text(f"Context Window: {params['context']}")
    st.sidebar.text(f"Embedding Dim: {params['embed_dim']}")
    st.sidebar.text(f"Activation: ReLU")
    st.sidebar.text(f"Vocab Size: {len(stoi)}")

    # --- Generation Controls ---
    st.sidebar.subheader("Generation Settings")
    seed = st.sidebar.number_input("Random Seed", value=1337) [cite: 47]
    temperature = st.sidebar.slider("Temperature (Randomness)", 0.1, 2.0, 0.8, 0.1) [cite: 48]
    top_k = st.sidebar.slider("Top-K Sampling (Focus)", 1, 50, 10, 1)
    num_to_gen = st.sidebar.number_input("Words to Generate (k)", 1, 100, 20) 

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Main Page ---
    st.info("Enter some starting text. Words not in the vocabulary will be treated as '<unk>'. [cite: 49]")
    
    # Set default text based on model
    default_text = "to be or not to" if 'cat1' in model_path.name else "if ( file is"
    user_input = st.text_area("Enter your context text:", default_text, height=150)

    if st.button("Generate"):
        # Tokenize user input
        if 'cat1' in model_path.name:
            # Natural language: simple split
            tokens = user_input.lower().split()
        else:
            # Code: find all non-space chunks
            tokens = re.findall(r'[^\s]+', user_input.lower())
        
        if not tokens:
            st.warning("Please enter some text to start.")
        else:
            with st.spinner(f"Generating {num_to_gen} words..."):
                generated_text = generate_sequence(
                    model, stoi, itos, tokens,
                    num_to_gen, temperature, top_k, params
                )
            
            st.subheader("Generated Text")
            # Display the full sequence (input + generated)
            st.markdown(f"**{user_input}** <span style='color: #0078D4; font-weight: bold;'>{generated_text}</span>", unsafe_allow_html=True)
else:
    st.error("Please place your trained `cat1_model.pt` and `cat2_model.pt` files in a folder named `q1_artifacts` and restart the app.")
