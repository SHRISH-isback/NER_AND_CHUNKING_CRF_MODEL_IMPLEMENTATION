# TRAINING AND TESTING STEPS

# Named Entity Recognition (NER) Model:

# Training the NER model:
- python ner_model.py --train train.csv --model ner_model.pkl

# Testing the NER model on your manually annotated test data:
- python ner_model.py --test ner_annotations.txt --model ner_model.pkl --output ner_predictions.txt

# Chunking Model:

# Training the chunking model:
- python chunking_model.py --train train.txt --model chunking_model.pkl

# Testing the chunking model on your manually annotated test data:
- python chunking_model.py --test chunk_annotations.txt --model chunking_model.pkl --output chunking_predictions.txt


# EVALUATION:

# Evaluating the chunking model and NER model:
- python eval.py ner_annotations.txt ner_predictions.txt chunk_annotations.txt chunking_predictions.txt

------------------------------------------------------------------------------------------------

# Model Architecture and Data Preprocessing

## Named Entity Recognition (NER) Model

### Hidden Markov Model (HMM) Architecture

- **States:** The hidden states represent NER tags (`B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-MISC`, `I-MISC`, `O`)
- **Observations:** The observed tokens in the text sequence
- **Components:**
  - **Transition Matrix:** Captures tag-to-tag transition probabilities (e.g., probability of transitioning from `B-PER` to `I-PER`)
  - **Emission Matrix:** Models word-to-tag emission probabilities (e.g., probability of observing "Gandhi" given state `B-PER`)
  - **Initial State Distribution:** Probabilities of starting with each tag
- **Inference:** Viterbi algorithm to find the most likely sequence of tags given the observed words

### Feature Engineering for NER

- Word-level features: lowercase word, word shape (capitalization, digits)
- Context features: previous and next words in a window of ±2
- Prefix and suffix features (character n-grams of length 2–4)
- Part-of-speech tags as additional features
- Binary features for capitalization patterns and digit presence

---

## Chunking Model

### Conditional Random Field (CRF) Architecture

- **Sequential Discriminative Model:** Directly models the conditional probability `P(tags|words)`
- **Linear-chain CRF:** Models dependencies between adjacent tags while considering the entire observation sequence
- **Components:**
  - **Feature Functions:** Capture relationships between observations and labels, and between adjacent labels
  - **Weight Parameters:** Learned during training to maximize the conditional likelihood of training data
- **Inference:** Forward-backward algorithm for training, Viterbi algorithm for decoding

### Feature Engineering for Chunking

- POS tags as primary features (crucial for chunking)
- Word identity features
- Context word and POS features in a window of ±2
- Chunk tag history (previous tags)
- Syntactic features like dependency relations (if available)
- Orthographic features (capitalization, special characters)

------------------------------------------------------------------------------------------

## Data Sources and Preprocessing Steps

### Data Sources

#### NER Task

- **Training Data:** CoNLL-2003 English NER dataset
  - Contains newspaper articles with `PER`, `ORG`, `LOC`, `MISC` annotations
  - 14,987 sentences with 203,621 tokens for training
  - Format: One token per line with tab-separated features and BIO-encoded NER tags

#### Chunking Task

- **Training Data:** CoNLL-2000 Chunking dataset
  - Contains Wall Street Journal text with various chunk types (`NP`, `VP`, `PP`, etc.)
  - 8,936 sentences and 211,727 tokens for training
  - Format: One token per line with word, POS tag, and BIO-encoded chunk tag

#### Testing Data

- Manually annotated text from Wikipedia
- Same text used for both NER and chunking evaluation
- Annotated according to assignment guidelines using BIO format

---

## Preprocessing Steps

### Data Cleaning and Formatting

1. **Sentence Segmentation:** Split text into sentences for processing
2. **Tokenization:** Break sentences into tokens following CoNLL conventions
3. **Format Conversion:** Convert data to appropriate format (one token per line, features tab-separated)
4. **Tag Mapping:** Map from ILMT tagset to CoNLL format according to assignment guidelines

### Feature Extraction

1. **Text Normalization:**
   - Lowercase transformation (with original case preserved as a feature)
   - Special character handling
   - Number normalization

2. **Feature Generation:**
   - Extract word n-grams (unigrams, bigrams)
   - Generate character n-grams for prefixes and suffixes
   - Create binary features for capitalization patterns
   - Part-of-speech tagging using NLTK’s Penn Treebank tagger
   - Word shape features (e.g., `"John"` → `"Xxxx"`)

3. **Feature Vectorization:**
   - Convert categorical features to numerical representations
   - Create sparse feature matrices for CRF training
   - Feature scaling where appropriate

### Train-Test Split

- Use entire CoNLL datasets for training
- Hold out manually annotated data exclusively for testing
- No cross-validation on test data to prevent contamination

### Data Augmentation (for improving model robustness)

- Entity substitution with similar entities from the same class
- Case variation for named entities
- Addition of noise words to test boundary detection

---

This architecture leverages the sequential nature of both NER and chunking tasks while providing robust feature engineering to capture lexical, syntactic, and contextual information essential for accurate predictions.
