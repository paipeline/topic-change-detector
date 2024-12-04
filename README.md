# Topic Change Detector

A tool that analyzes text to detect and visualize topic changes using BERT embeddings and LDA topic modeling. This application helps identify both major and minor shifts in topic throughout a piece of text, making it useful for AI content analysis, recording/document segmentation, text structure understanding.

## 🌟 Features

- **Real-time Analysis**: Instantly analyze text and visualize topic changes
- **Interactive Visualization**: 
  - Dynamic similarity graph with Plotly
  - Color-coded text highlighting
  - Detailed topic transition analysis
- **Dual Detection Levels**: 
  - Major topic changes (significant shifts)
  - Minor topic changes (subtle transitions)
- **Customizable Parameters**: Adjustable thresholds for change detection
- **Topic Extraction**: Keywords extraction for each text segment
- **Comprehensive Statistics**: Overview of topic changes and text structure

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **NLP Processing**:
  - BERT (Sentence Transformers)
  - LDA (Latent Dirichlet Allocation)
  - NLTK
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

## 🚀 Installation

1. **Clone the repository**
   ```bash
    git clone https://github.com/paipeline/topic-change-detector.git
    cd topic-change-detector```
2. **Create and activate virtual environment (recommended)**
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux 
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8501`
   - The FastAPI backend will be available at `http://localhost:8000`

3. **Analyze your text**
   - Paste or upload your text content
   - Adjust detection parameters if needed
   - View the analysis results and visualizations

## 🔧 Configuration

You can customize the detection parameters in `config.yaml`:

- `major_change_threshold`: Threshold for major topic changes - change deviation of topic from the main theme.
- `minor_change_threshold`: Threshold for subtle transitions


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
