# Emotionally Aware Dating Simulator Dialogue Chatbot

An emotionally intelligent dating simulator with a multi-component system to detect, track, and respond to emotional states while generating contextually appropriate dialogue.

## Architecture

The system consists of the following components:

1. **User Interface**: Text-based CLI for interaction
2. **Conversation Engine**: Manages input preprocessing and context
3. **Emotion Detection**: Sentiment analysis and emotion labeling
4. **Persona Engine**: Personality profiles and mood/state modeling
5. **Dialogue Generator**: RNN/Transformer-based response generation
6. **Emotional State Tracker**: Maintains persistent emotional context

## Project Structure

```
CS425_Project/
├── data/                      # Dataset storage
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Preprocessed data
│   └── scenarios/             # Generated scenarios
├── notebooks/                 # Jupyter notebooks for experimentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CS425_Project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required datasets (see Data section)

## Datasets
> Data to be placed in the `data/raw` folder
 
The project uses the following datasets:
- **Emotion Dataset** - 69k emotion-labeled text samples for emotion detection training
- **Love Is Blind Dataset** - TV show scripts from Season 1 (11 episodes) for dialogue generation and persona training
- **Processed Love Is Blind** - Cleaned and structured CSV format of the Love Is Blind scripts

### Downloading Datasets

Datasets can be accessed from the shared Google Drive folder:
**[CS425 Project Datasets](https://drive.google.com/drive/folders/15R-d4qYKBBvxhDfbBsFZSi6Y8V7L0zbz?usp=sharing)**

- [Visual Novels Dataset](https://huggingface.co/datasets/alpindale/visual-novels)
- [MELD Dataset](https://github.com/declare-lab/MELD)

### Finetuned Checkpoints
Finetuned checkpoints can be accessed from the shared Google Drive folder:
- https://drive.google.com/drive/folders/15R-d4qYKBBvxhDfbBsFZSi6Y8V7L0zbz?usp=sharing

## Usage

### Data Preprocessing and Finetuning

Download, load, preprocess, and fine tunetune the model by running the respective jupyter notebook in the  `notebooks` folder:
```bash
cd notebook
jupyter notebook
```

- `notebook/MELD`: MELD Dataset
- `notebook/VN_emotion`: Visual Novel Dataset with emotion classification
- `notebook/VN_no_emotion`: Visual Novel Dataset without emotion classification
- `notebook/VN_split`: Visual Novel Dataset with emotion and split characters

### Evaluation

In each notebook directory has their own respective evaluation notebook called `notebook/<variant>/04_evaluation_*.ipynb`.

There is also an comparison notebook comprised of all the visual novels trained and evaluated called `notebook/05_model_comparison__VN.ipynb`

### Running the Chatbot (Inference)

Launch the interactive CLI:
```bash
# List Arguments
python src/app/cli.py --help

# Example with our best model
python src/app/cli.py -m checkpoints/08_11_2025__cleaned_merged__128_len__12_epoch__30_eval/final -t merged
```

- Download or finetune your own model and place it in the `checkpoints` folder.