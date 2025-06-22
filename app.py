import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid

# --- Define the Generator Architecture ---
# NOTE: This must be the *exact* same architecture as in your training script.
latent_dim = 100
n_classes = 10
img_size = 28
channels = 1

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_size * img_size * channels),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), channels, img_size, img_size)
        return img

# --- Load the Trained Model ---
@st.cache_resource
def load_model():
    model = Generator()
    # Load the model state dict, mapping to CPU
    # This is crucial for Streamlit Cloud's free tier, which is CPU-based
    model.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

generator = load_model()
device = torch.device('cpu') # Ensure everything runs on CPU
generator.to(device)


# --- Streamlit Web App Interface ---

st.set_page_config(layout="wide", page_title="Handwritten Digit Generator")

st.title("✍️ Handwritten Digit Generator")
st.write("Generate images of handwritten digits (0-9) using a Conditional GAN trained on the MNIST dataset.")

# --- User Input and Generation ---

st.sidebar.header("Controls")
selected_digit = st.sidebar.selectbox("Select a digit to generate", options=list(range(10)))

# Generate 5 images
num_images = 5

if st.sidebar.button("Generate Images", type="primary"):

    with st.spinner("Generating digits..."):
        # Generate latent vectors (noise)
        z = torch.randn(num_images, latent_dim, device=device)

        # Generate labels for the selected digit
        labels = torch.LongTensor([selected_digit] * num_images).to(device)

        # Generate images
        with torch.no_grad():
            generated_imgs = generator(z, labels)

        # Post-process for display
        # Un-normalize from [-1, 1] to [0, 1]
        generated_imgs = 0.5 * generated_imgs + 0.5

        # Create a grid and convert to displayable format
        grid = make_grid(generated_imgs, nrow=5, normalize=True)
        img_grid = np.transpose(grid.cpu().numpy(), (1, 2, 0))

        st.subheader(f"Generated Images for Digit: {selected_digit}")
        st.image(img_grid, width=500)

else:
    st.info("Select a digit from the sidebar and click 'Generate Images'.")