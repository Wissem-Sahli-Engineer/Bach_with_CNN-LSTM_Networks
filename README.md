# 🎼 Bach AI: Deep Polyphonic Music Generation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent deep learning system that generates **4-voice polyphonic classical chorales in the style of Johann Sebastian Bach**. It combines **dilated causal 1D convolutions** with **Long Short-Term Memory (LSTM)** recurrent networks to model complex multi-voice harmonies and synthesize them into playable WAV audio.

---

## 🎧 Audio Samples

Listen to the comparison between the original ground-truth Bach chorale and the AI-generated composition:

| Sample Type | Description | Audio File Link |
| :--- | :--- | :--- |
| **Original Bach Chorale** | Ground-truth 4-voice chorale snippet from the JSB test dataset | 🔊 [Listen to Original Bach](file:///Users/wess/Desktop/Bach_with_RNN_CNN/bach_test_4.wav) |
| **AI-Generated Chorale** | Autoregressively generated polyphonic composition ($T=1.0$) | 🎼 [Listen to AI Generated Bach](file:///Users/wess/Desktop/Bach_with_RNN_CNN/bach_medium.wav) |

---

## 🏗️ Architecture & Flow Scheme

```mermaid
graph LR
    subgraph Pipeline["Data Pipeline"]
        A["📄 JSB CSVs"] --> B["⚙️ Shift & Flatten"]
        B --> C["🪟 tf.data Windowing"]
        C --> D["🔀 Batch & Slice X/Y"]
    end

    subgraph Model["Hybrid CNN-LSTM Model"]
        D --> E["🔤 Embedding Layer"]
        E --> F["🌊 Dilated Conv1D Blocks<br/>(Dilation rates: 1, 2, 4, 8)"]
        F --> G["🧠 Recurrent LSTM Layer"]
        G --> H["🎯 Softmax Classifier"]
    end

    subgraph Synthesis["Audio DSP Engine"]
        H --> I["🎲 Temperature Sampling"]
        I --> J["🎵 Pitch to Frequency"]
        J --> K["🔊 Sine Wave Synthesizer"]
        K --> L["💾 WAV Audio Output"]
    end
```

---

## 📂 Directory Structure

```text
Bach_with_RNN_CNN/
│
├── 📄 README.md                        # Project documentation
├── 📄 requirements.txt                 # Dependencies list
├── 📜 utils.py                         # Data processing & audio DSP module
├── 📓 main.ipynb                       # Training & composition notebook
│
├── 📁 data/                            # Dataset files
│   └── 📁 jsb_chorales/                # Processed splits (train, valid, test)
│
├── 📁 models/                          # Trained model artifacts
│   └── 💾 my_bach_model.keras          # Saved Keras hybrid model
│
└── 🎵 Audio Samples:
    ├── 🎧 bach_medium.wav              # AI Generated sample (T = 1.0)
    ├── 🎧 bach_cold.wav                # AI Generated sample (T = 0.8)
    ├── 🎧 bach_hot.wav                 # AI Generated sample (T = 1.5)
    └── 🎧 bach_test_4.wav              # Ground-truth original sample
```

---

## 📄 File Details

* [main.ipynb](file:///Users/wess/Desktop/Bach_with_RNN_CNN/main.ipynb): Main execution notebook for dataset extraction, EDA, neural network training, evaluation, and interactive composition.
* [utils.py](file:///Users/wess/Desktop/Bach_with_RNN_CNN/utils.py): Core utility library providing:
  * Data pipeline transformers ([`load_data`](file:///Users/wess/Desktop/Bach_with_RNN_CNN/utils.py#L14), [`bach_dataset`](file:///Users/wess/Desktop/Bach_with_RNN_CNN/utils.py#L36), [`preprocess`](file:///Users/wess/Desktop/Bach_with_RNN_CNN/utils.py#L32))
  * Sequence generators ([`generate_chorale`](file:///Users/wess/Desktop/Bach_with_RNN_CNN/utils.py#L105), [`generate_chorale_v2`](file:///Users/wess/Desktop/Bach_with_RNN_CNN/utils.py#L115))
  * DSP audio synthesizer ([`notes_to_frequencies`](file:///Users/wess/Desktop/Bach_with_RNN_CNN/utils.py#L67), [`frequencies_to_samples`](file:///Users/wess/Desktop/Bach_with_RNN_CNN/utils.py#L72), [`play_chords`](file:///Users/wess/Desktop/Bach_with_RNN_CNN/utils.py#L94))
* [models/my_bach_model.keras](file:///Users/wess/Desktop/Bach_with_RNN_CNN/models/my_bach_model.keras): Trained model weights checkpoint.
* [requirements.txt](file:///Users/wess/Desktop/Bach_with_RNN_CNN/requirements.txt): Environment requirement specifications.
* [data/](file:///Users/wess/Desktop/Bach_with_RNN_CNN/data): Directory storing the JSB Chorales dataset CSV split partitions.

---

## 🧮 How It Works (Core Logic)

### 1. Data Representation & Index Shifting

MIDI pitch numbers range from $36$ to $81$, where $0$ denotes rests/silence. Pitch values are normalized using zero-relative indexing based on the minimum note ($n_{\text{min}} = 36$):

$$n_{\text{shifted}} = \begin{cases} 0 & \text{if } n = 0 \\ n - n_{\text{min}} + 1 & \text{if } n > 0 \end{cases}$$

This compacts the active vocabulary size to $N = 47$. Four-part chorale voice matrices are flattened into sequential arpeggio streams $[S_t, A_t, T_t, B_t, S_{t+1}, \dots]$ for step-by-step prediction.

### 2. Dilated Convolutions & Receptive Fields

1D causal convolutions employ doubling dilation rates ($d \in \{1, 2, 4, 8\}$) to capture rhythmic patterns across time steps without future context leakage:

$$(y *_{d} k)(t) = \sum_{m=0}^{K-1} k(m) \cdot x(t - d \cdot m)$$

### 3. Temperature-Scaled Sampling

Logits $z_k$ from the Softmax layer are scaled by temperature parameter $T$:

$$P(y_t = k) = \frac{\exp(z_k / T)}{\sum_{j=1}^{N} \exp(z_j / T)}$$

* **Low Temperature ($T = 0.8$)**: Conservative, strict classical rules.
* **High Temperature ($T = 1.5$)**: Experimental, creative chromatic variations.

### 4. Waveform Audio Synthesis

MIDI notes are converted to frequencies (Hz):

$$f = 440 \times 2^{\frac{n - 69}{12}}$$

Sine wave signals are synthesized per voice, phase-aligned at beat boundaries, and merged with a quadratic fade-out envelope.

---

## 🛠️ Setup & Requirements

### Installation

1. **Navigate to project directory**:
   ```bash
   cd /Users/wess/Desktop/Bach_with_RNN_CNN
   ```

2. **Activate Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install tensorflow pandas numpy scipy ipython notebook
   ```

---

## 🎮 Controls & Usage

Run all cells in [main.ipynb](file:///Users/wess/Desktop/Bach_with_RNN_CNN/main.ipynb) or generate music programmatically:

```python
import tensorflow as tf
from utils import generate_chorale_v2, play_chords

# Load pre-trained model
model = tf.keras.models.load_model("models/my_bach_model.keras")

# Generate new 4-voice composition (56 chords)
new_chorale = generate_chorale_v2(model, seed_chords=seed, length=56, temperature=1.0)

# Synthesize and export WAV
play_chords(new_chorale, tempo=160, filepath="generated_bach.wav")
```
