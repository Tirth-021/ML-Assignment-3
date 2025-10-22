import streamlit as st

@st.cache_resource
def load_model(model_path):
    """
    Loads the model and vocab. This is cached so it only runs once.
    """
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Recreate the exact model architecture
        vocab_size = len(checkpoint['itos'])
        
        # ASSUMPTION: We must hard-code the params for each model
        # This resolves the assignment contradiction.
        if 'cat1' in model_path.name:
            model_params = {
                'context': DEFAULT_CONTEXT,
                'embed_dim': DEFAULT_EMBED_DIM,
                'h1': 1024,
                'h2': 512,
                'dropout': 0.4
            }
        elif 'cat2' in model_path.name:
            model_params = {
                'context': DEFAULT_CONTEXT,
                'embed_dim': DEFAULT_EMBED_DIM,
                'h1': 1024,
                'h2': 512,
                'dropout': 0.4
            }
        else:
            # Fallback for other models
            model_params = {
                'context': DEFAULT_CONTEXT,
                'embed_dim': DEFAULT_EMBED_DIM,
                'h1': 512,
                'h2': 256,
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
        st.error(f"Model file not found at {model_path}. Did you create the 'q1_artifacts' folder and add your .pt files?")
        return None, None, None, None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None, None, None, None

# ----------------------------------------------------------------------
# 4. Streamlit UI
# ----------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("Next-Word Prediction with MLP")

# --- Sidebar Controls ---
st.sidebar.title("Controls")

ARTIFACTS_DIR = '/kaggle/working/q1_artifacts'

model_options = {
    "Category 1: Natural (Shakespeare)": f'{ARTIFACTS_DIR} / cat1_model.pt',
    "Category 2: Structured (Linux C Code)": f'{ARTIFACTS_DIR} / cat2_model.pt'
}
model_choice = st.sidebar.selectbox("Choose Model Variant", model_options.keys())
model_path = model_options[model_choice]

# Load the selected model
model, stoi, itos, params = load_model(model_path)

if model:
    # Display the loaded model's parameters (as required by assignment)
    st.sidebar.subheader("Loaded Model Parameters")
    st.sidebar.text(f"Context Window: {params['context']}")
    st.sidebar.text(f"Embedding Dim: {params['embed_dim']}")
    st.sidebar.text(f"Hidden 1: {params['h1']}")
    st.sidebar.text(f"Hidden 2: {params['h2']}")
    st.sidebar.text(f"Activation: ReLU")
    st.sidebar.text(f"Vocab Size: {len(stoi)}")

    # --- Generation Controls ---
    st.sidebar.subheader("Generation Settings")
    seed = st.sidebar.number_input("Random Seed", value=1337)
    temperature = st.sidebar.slider("Temperature (Randomness)", 0.1, 2.0, 0.8, 0.1)
    top_k = st.sidebar.slider("Top-K Sampling (Focus)", 1, 50, 10, 1)
    num_to_gen = st.sidebar.number_input("Words to Generate", 1, 100, 20)

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Main Page ---
    st.info("Enter some starting text. The model will generate the next words autoregressively.")
    
    # Set default text based on model
    default_text = "Woe, the" if 'cat1' in model_path.name else "int main ( ) {"
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
