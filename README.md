# Natural Language Processing - Bachelor Course

## Course Overview

This repository contains materials, code examples, and exercises for the Natural Language Processing module at Fachhochschule Südwestfalen. The course covers fundamental and advanced NLP techniques using modern Python libraries and frameworks.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository** (or download as ZIP):
   ```bash
   git clone <repository-url>
   cd NLP_BA2526
   ```

2. **Create a virtual environment**:
   ```bash
   # On Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download language models**:
   ```bash
   # Download NLTK data
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   
   # Download spaCy German model
   python -m spacy download de_core_news_sm
   ```

5. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

## Topics Covered

### 0. Text Processing Fundamentals with Regular Expressions
- **Overview**: Introduction to regex, pattern matching, text cleaning, and basic text processing techniques
- **Key Libraries**: re (Python built-in), string, unicodedata
- **Learning Objectives**: Master regex patterns, build text cleaning pipelines, extract information from text

### 1. Introduction to NLP and Text Processing
- **Overview**: Fundamentals of natural language processing, text preprocessing, tokenization, and basic text analysis
- **Key Libraries**: NLTK, spaCy
- **Learning Objectives**: Understand basic NLP concepts, text preprocessing pipelines, and tokenization methods

### 2. Text Classification with Traditional Methods
- **Overview**: Introduction to text classification using traditional machine learning approaches
- **Key Concepts**: Feature extraction (TF-IDF, Bag of Words), classification algorithms (SVM, Naive Bayes)
- **Learning Objectives**: Build text classifiers using scikit-learn, understand feature engineering for text

### 3. Word Embeddings and Vector Representations
- **Overview**: Word2Vec, GloVe, and FastText for creating word representations
- **Key Libraries**: Gensim, spaCy, Hugging Face Transformers
- **Learning Objectives**: Understand vector space models, train custom word embeddings, use pre-trained embeddings

### 4. Advanced Text Classification with BERT
- **Overview**: Using transformer models for text classification tasks
- **Key Libraries**: Hugging Face Transformers, PyTorch/TensorFlow
- **Learning Objectives**: Fine-tune BERT models, understand attention mechanisms, implement state-of-the-art classifiers

### 5. Named Entity Recognition (NER)
- **Overview**: Identifying and classifying named entities in text
- **Key Libraries**: spaCy, Hugging Face Transformers
- **Learning Objectives**: Use pre-trained NER models, train custom NER systems, evaluate NER performance

### 6. Topic Modeling
- **Overview**: Discovering topics in document collections
- **Key Methods**: LDA, Top2Vec, BERTopic
- **Learning Objectives**: Apply topic modeling techniques, interpret topic models, visualize results

### 7. Automatic Speech Recognition (ASR)
- **Overview**: Converting speech to text using deep learning models
- **Key Libraries**: SpeechRecognition, Wav2Vec2, Whisper
- **Learning Objectives**: Implement ASR systems, understand audio preprocessing, evaluate ASR performance

### 8. Conversational AI with RASA
- **Overview**: Building chatbots and conversational interfaces
- **Key Framework**: RASA
- **Learning Objectives**: Design conversation flows, implement intent recognition, deploy chatbots

### 9. German NLP and GermEval Tasks
- **Overview**: Working with German language datasets and shared tasks
- **Key Datasets**: GermEval datasets for sentiment analysis, offensive language detection
- **Learning Objectives**: Handle language-specific challenges, participate in shared tasks

## Exercise Collection

### Exercise 0: Regular Expressions and Text Basics
**Difficulty**: Beginner  
**Estimated Time**: 2-3 hours

**Learning Objectives**:
1. **Pattern Matching**: Master regex patterns for German text processing
2. **Text Cleaning**: Build robust text preprocessing pipelines
3. **Data Extraction**: Extract structured information from unstructured text
4. **Validation**: Implement data validation using regex patterns
5. **Text Analysis**: Perform basic statistical analysis of text data

**What You'll Build**:
- German text cleaning pipeline
- Email/phone number extraction system
- Text validation framework
- Basic text statistics analyzer

**Required Libraries**: `re`, `string`, `unicodedata`, `matplotlib`, `pandas`

---

### Exercise 1: Introduction to NLP and Text Processing
**Difficulty**: Beginner  
**Estimated Time**: 3-4 hours

**Learning Objectives**:
1. **NLP Fundamentals**: Understand core NLP concepts and terminology
2. **Text Preprocessing**: Master tokenization, normalization, and cleaning
3. **German NLP**: Handle German language specifics (umlauts, compound words)
4. **Statistical Analysis**: Calculate and interpret text statistics
5. **Visualization**: Create meaningful text data visualizations
6. **Library Integration**: Work with NLTK and spaCy for German text

**What You'll Build**:
- Multi-method tokenization system
- German text statistics analyzer
- Sentence segmentation tool
- Text complexity assessment framework

**Required Libraries**: `nltk`, `spacy`, `matplotlib`, `pandas`, `wordcloud`

---

### Exercise 2: Traditional Text Classification
**Difficulty**: Beginner-Intermediate  
**Estimated Time**: 4-5 hours

**Learning Objectives**:
1. **Feature Engineering**: Convert text to numerical features (TF-IDF, Count Vectors)
2. **Dataset Creation**: Build and prepare text classification datasets
3. **Model Training**: Train traditional ML classifiers (Naive Bayes, SVM, Logistic Regression)
4. **Model Evaluation**: Assess classifier performance using metrics and cross-validation
5. **German Text Classification**: Handle German language specifics
6. **Pipeline Creation**: Build complete classification pipelines

**What You'll Build**:
- German sentiment analysis system
- Multi-class text classifier
- Feature comparison tools
- Model evaluation framework
- Production-ready classification pipeline

**Required Libraries**: `scikit-learn`, `pandas`, `seaborn`, `matplotlib`

---

### Exercise 3: Word Embeddings and Vector Representations
**Difficulty**: Intermediate  
**Estimated Time**: 4-5 hours

**Learning Objectives**:
1. **Vector Space Models**: Understand word embeddings and semantic spaces
2. **Pre-trained Embeddings**: Work with German Word2Vec and FastText models
3. **Custom Training**: Train Word2Vec models on domain-specific corpora
4. **Similarity Analysis**: Compute semantic similarities and analogies
5. **Visualization**: Create 2D/3D visualizations of word embeddings
6. **Evaluation**: Assess embedding quality using intrinsic and extrinsic methods

**What You'll Build**:
- German word similarity analyzer
- Custom Word2Vec training pipeline
- Interactive embedding visualizer
- Analogy solver system
- Embedding evaluation framework

**Required Libraries**: `gensim`, `sklearn`, `matplotlib`, `seaborn`, `plotly`

---

### Exercise 4: BERT Classification
**Difficulty**: Intermediate-Advanced  
**Estimated Time**: 5-6 hours

**Learning Objectives**:
1. **BERT Architecture**: Understand transformer-based language models and attention mechanisms
2. **German BERT Models**: Work with pre-trained German BERT variants (GBERT, DistilBERT)
3. **Fine-tuning Process**: Adapt pre-trained models for specific classification tasks
4. **Tokenization**: Handle BERT's WordPiece tokenization for German text
5. **Performance Comparison**: Compare BERT with traditional ML approaches
6. **Model Evaluation**: Assess BERT model performance with appropriate metrics

**What You'll Build**:
- German sentiment classifier using BERT
- Performance comparison framework
- BERT model fine-tuning pipeline
- Attention visualization system
- Production-ready classification API

**Required Libraries**: `transformers`, `torch`, `datasets`, `matplotlib`, `seaborn`

---

### Exercise 5: Named Entity Recognition
**Difficulty**: Intermediate  
**Estimated Time**: 4-5 hours

**Learning Objectives**:
1. **NER Fundamentals**: Understand named entity types and recognition challenges
2. **German NER**: Work with German-specific entity types and language patterns
3. **Multi-Model Comparison**: Compare spaCy, Flair, and BERT-based NER systems
4. **Custom NER**: Build and train custom NER models for specific domains
5. **Evaluation Methods**: Assess NER performance using appropriate metrics
6. **Entity Analysis**: Extract insights from identified entities

**What You'll Build**:
- Multi-model German NER system
- Custom entity extraction pipeline
- Entity relationship visualizer
- NER performance comparison tool
- Domain-specific NER trainer

**Required Libraries**: `spacy`, `flair`, `transformers`, `pandas`, `matplotlib`, `seaborn`

**Required Libraries**: `spacy`, `pandas`, `networkx`, `plotly`

---

### Exercise 6: Topic Modeling
**Difficulty**: Intermediate  
**Estimated Time**: 4-5 hours

**Learning Objectives**:
1. **Topic Modeling Fundamentals**: Understand unsupervised topic discovery
2. **LDA Implementation**: Build and optimize Latent Dirichlet Allocation models
3. **German Text Processing**: Handle German-specific preprocessing for topic modeling
4. **Model Evaluation**: Assess topic coherence and model quality
5. **Visualization**: Create interactive topic model visualizations
6. **Practical Applications**: Apply topic modeling to real-world German text data

**What You'll Build**:
- Comprehensive German topic modeling system
- LDA model optimization pipeline
- Interactive topic visualization dashboard
- Document-topic analysis tools
- Topic model evaluation framework

**Required Libraries**: `gensim`, `pyldavis`, `sklearn`, `matplotlib`, `plotly`

---

### Exercise 7: Speech Recognition
**Difficulty**: Intermediate-Advanced  
**Estimated Time**: 5-6 hours

**Learning Objectives**:
1. **ASR Fundamentals**: Understand automatic speech recognition principles
2. **Audio Processing**: Master audio preprocessing and feature extraction
3. **German ASR**: Work with German speech recognition models and datasets
4. **Model Comparison**: Compare different ASR approaches (Wav2Vec2, Whisper, cloud APIs)
5. **Performance Evaluation**: Assess ASR accuracy using appropriate metrics
6. **Real-time Processing**: Implement streaming speech recognition systems

**What You'll Build**:
- Comprehensive German ASR system
- Audio preprocessing pipeline  
- Multi-model ASR comparison tool
- Speech quality assessment framework
- Real-time transcription interface

**Required Libraries**: `transformers`, `librosa`, `soundfile`, `speechrecognition`, `whisper`

---

### Exercise 8: RASA Chatbot
**Difficulty**: Advanced  
**Estimated Time**: 6-7 hours

**Learning Objectives**:
1. **Conversational AI**: Understand dialogue systems and chatbot architecture
2. **Intent Recognition**: Build robust intent classification for German language
3. **Entity Extraction**: Implement named entity recognition for conversational context
4. **Dialogue Management**: Design and implement conversation flows
5. **RASA Framework**: Master RASA's components (NLU, Core, Actions)
6. **Production Deployment**: Deploy chatbots for real-world usage

**What You'll Build**:
- Complete German university chatbot
- Comprehensive NLU training data
- Custom action server with API integrations
- Interactive conversation interface
- Chatbot evaluation and testing framework

**Required Libraries**: `rasa`, `rasa-sdk`, `pyyaml`, `requests`

---

### Exercise 9: Offensive Language Detection
**Difficulty**: Intermediate-Advanced  
**Estimated Time**: 5-6 hours

**Learning Objectives**:
1. **Content Moderation**: Understand challenges in automated content moderation
2. **German Offensive Language**: Work with German profanity and hate speech detection
3. **Ethical AI**: Address bias, fairness, and ethical considerations in NLP
4. **Advanced Classification**: Handle imbalanced datasets and multi-label classification
5. **Model Evaluation**: Assess sensitive NLP applications with appropriate metrics
6. **Production Systems**: Build robust content moderation systems

**What You'll Build**:
- German offensive language detection system
- Ethical AI evaluation framework
- Bias analysis and mitigation tools
- Content moderation API
- Comprehensive evaluation dashboard

**Required Libraries**: `transformers`, `torch`, `sklearn`, `matplotlib`, `seaborn`

---

## Total Course Duration
**Estimated Total Time**: 42-50 hours  
**Recommended Schedule**: 10-12 weeks (4-5 hours per week)  
**Prerequisites**: Basic Python programming, linear algebra fundamentals

## Assessment Guidelines

### Evaluation Criteria
- **Code Quality** (25%): Clean, documented, and efficient code
- **Technical Implementation** (30%): Correct use of NLP techniques and libraries
- **Analysis and Interpretation** (25%): Insightful analysis of results
- **Documentation** (20%): Clear explanations and reporting

### Submission Requirements
- Complete source code with comments
- Jupyter notebooks with explanations
- Written report (2-3 pages per exercise)
- Presentation slides for final projects

## Resources and References

### Essential Libraries
```bash
# Install all dependencies with:
pip install -r requirements.txt

# Post-installation commands:
python -m spacy download de_core_news_sm
python -m spacy download de_core_news_md  
python -m nltk.downloader punkt stopwords
```

### Library Overview
- **Core NLP**: `nltk`, `spacy`, `transformers`, `datasets`
- **Machine Learning**: `scikit-learn`, `torch`, `tensorflow`
- **Audio Processing**: `librosa`, `soundfile`, `speechrecognition`, `whisper`
- **Conversational AI**: `rasa`, `rasa-sdk`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`, `wordcloud`
- **Specialized**: `gensim`, `flair`, `evaluate`

### Recommended Datasets
- [German Sentiment Classification](https://www.aclweb.org/anthology/L18-1636/)
- [GermEval Shared Tasks](https://germeval.org/)
- [Project Gutenberg German Texts](https://www.gutenberg.org/)
- [Common Voice German](https://commonvoice.mozilla.org/)

### Additional Reading
- [Speech and Language Processing - Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/)
- [Natural Language Processing with Python - NLTK Book](https://www.nltk.org/book/)
- [Hugging Face Course](https://huggingface.co/course)
- [spaCy Documentation](https://spacy.io/usage)

## Getting Started

1. **Environment Setup**:
   ```bash
   git clone [repository-url]
   cd nlp-course
   pip install -r requirements.txt
   python -m spacy download de_core_news_sm
   ```

2. **Download NLTK Data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

3. **Start with Exercise 1** and progress through the exercises systematically

4. **Join Course Discussion**: Use the course forum for questions and collaboration

## Support and Office Hours

- **Office Hours**: Tuesdays 14:00-16:00, Room XXX
- **Email**: [instructor-email]
- **Course Forum**: [forum-link]
- **Technical Issues**: Create GitHub issues in this repository

---

*Last Updated: October 2025*  
*Course Instructor: [Instructor Name]*  
*Fachhochschule Südwestfalen - Fachbereich Informatik*