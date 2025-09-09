# YouTube Video Q&A with RAG

This application allows you to ask questions about YouTube videos using Retrieval-Augmented Generation (RAG). It extracts the video's transcript, processes it, and creates a question-answering system powered by OpenAI's GPT models.

## Features

- Extract transcripts from YouTube videos
- Process and chunk the transcript for efficient retrieval
- Use FAISS for efficient similarity search
- Answer questions about the video content using OpenAI's GPT models
- Simple and intuitive web interface using Streamlit

## Prerequisites

- Python 3.8+
- OpenAI API key
- FFmpeg (for audio processing)

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run youtube_rag.py
   ```
2. Open the provided URL in your web browser
3. Enter a YouTube video URL in the sidebar
4. Once the video is processed, you can start asking questions about its content

## How It Works

1. The application extracts the video ID from the YouTube URL
2. It retrieves the video's transcript using the YouTube Transcript API
3. The transcript is split into smaller chunks for processing
4. These chunks are embedded using HuggingFace's sentence transformers
5. A FAISS vector store is created for efficient similarity search
6. When you ask a question, the system retrieves the most relevant chunks and uses GPT to generate an answer

## Notes

- The application works best with videos that have accurate captions
- Longer videos may take some time to process
- The quality of answers depends on the quality of the video's captions

## License

This project is licensed under the MIT License.
