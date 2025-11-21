# Emotionally Aware Dating Simulator Dialogue Chatbot

An emotionally intelligent dating simulator with a multi-component system to detect, track, and respond to emotional states while generating contextually appropriate dialogue.

## Project Overview

This project develops a dating simulator chatbot with the following capabilities:
- **Emotion Detection**: Classifies user emotions using fine-tuned transformer models
- **Persona Engine**: Maintains a consistent personality (Prof Gao archetype)
- **Dialogue Generation**: Generates contextually appropriate, emotion-aware responses
- **Emotional State Tracker**: Tracks conversation history and affection levels

## Features

- Emotional responses to users based on detected sentiment
- Dynamic affection level tracking based on conversation context
- Single persona model trained for consistent personality
- Multi-component architecture for modularity and extensibility

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

The project uses the following datasets:
- **Emotion Dataset** - 69k emotion-labeled text samples for emotion detection training
- **Love Is Blind Dataset** - TV show scripts from Season 1 (11 episodes) for dialogue generation and persona training
- **Processed Love Is Blind** - Cleaned and structured CSV format of the Love Is Blind scripts

### Downloading Datasets

Datasets can be accessed from the shared Google Drive folder:
📁 **[CS425 Project Datasets](https://drive.google.com/drive/u/2/folders/1qn5ori_X3XGUCwJ1qM3cmhweeSH_stM-)**

## Configuration

Edit the YAML files in the `config/` directory to customize:
- `model_config.yaml`: Model architectures and hyperparameters
- `training_config.yaml`: Training settings and optimization
- `data_config.yaml`: Dataset paths and preprocessing options

## Usage

### Training

Train the emotion detection model:
```bash
python scripts/train.py --model emotion --config config/training_config.yaml
```

Train the dialogue generation model:
```bash
python scripts/train.py --model dialogue --config config/training_config.yaml
```

### Evaluation

Evaluate model performance:
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### Running the Chatbot

Launch the interactive CLI:
```bash
python scripts/run_chatbot.py
```

## Development Workflow

1. **Experimentation**: Use Jupyter notebooks in `notebooks/` for exploration
2. **Implementation**: Modularize code in `src/` modules
3. **Training**: Use scripts in `scripts/` for training
4. **Deployment**: Run the final chatbot using `run_chatbot.py`

## Implementation Phases

### Phase 1: Data Acquisition & Preprocessing
- Load and combine datasets
- Implement emotion label mapping
- Apply data augmentation
- Generate dating scenarios

### Phase 2: Emotion Detection Model
- Fine-tune BERT/RoBERTa models
- Train emotion classifier
- Evaluate on validation set

### Phase 3: Dialogue Generation & Persona Engine
- Implement dialogue generation with LoRA
- Define persona characteristics
- Build emotional state tracker

### Phase 4: Integration & Joint Training
- Integrate all components
- Optimize end-to-end system
- Balance emotion accuracy and response quality

### Phase 5: Evaluation & Feedback
- Measure performance metrics
- Gather qualitative feedback
- Iterate on improvements

## Evaluation Metrics

- **Emotion Classification**: F1-score, precision, recall
- **Dialogue Quality**: BLEU, perplexity
- **Emotional Consistency**: Temporal coherence
- **User Engagement**: Conversation length, appropriateness ratings

## Technologies

- **Language**: Python 3.8+
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Optimization**: LoRA (PEFT library)
- **Data Processing**: NLTK, spaCy, scikit-learn
- **Development**: Jupyter, Git

## Team

CS425 G2T3 Conversational AI

## License

Academic project for CS425 course.

## References

- EmpatheticDialogues Dataset
- GoEmotions Dataset
- LoRA: Low-Rank Adaptation of Large Language Models
- DialogueLLM: Context and Emotion Knowledge-Tuned LLMs
