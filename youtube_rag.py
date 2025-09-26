import os
import streamlit as st
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import requests
import base64
import io

def get_video_id(url):
    """Extract video ID from YouTube URL."""
    try:
        if 'youtu.be' in url:
            return url.split('/')[-1].split('?')[0]
        elif 'youtube.com' in url and 'v=' in url:
            video_id = url.split('v=')[1].split('&')[0]
            return video_id
        elif 'youtube.com' in url and '/watch' in url:
            # Handle cases where v= might not be present
            parts = url.split('/')
            for part in parts:
                if len(part) == 11 and part.isalnum():
                    return part
        # Handle URLs with query parameters
        import re
        pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return None
    except:
        return None

def get_elevenlabs_voices():
    """Get list of available voices from ElevenLabs API."""
    try:
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not elevenlabs_api_key:
            return []
            
        url = "https://api.elevenlabs.io/v1/voices"
        headers = {
            "Accept": "application/json",
            "xi-api-key": elevenlabs_api_key
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            voices = response.json().get("voices", [])
            return [(voice["voice_id"], voice["name"]) for voice in voices]
        else:
            return []
    except Exception as e:
        st.error(f"Error fetching voices: {str(e)}")
        return []

def text_to_speech_elevenlabs(text, voice_id=None):
    """Convert text to speech using ElevenLabs API with custom voice."""
    try:
        # Get ElevenLabs API key from environment
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not elevenlabs_api_key:
            return None
        
        # Use default voice if none specified
        if not voice_id:
            voice_id = "qpY4eaUGeCWHEUIVdbGM"  # User's custom voice
            
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        # Debug: Log the request
        print(f"Making TTS request to: {url}")
        print(f"Using voice_id: {voice_id}")
        print(f"Text length: {len(text)}")
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

def upload_voice_sample():
    """Upload voice sample to create custom voice clone."""
    st.subheader("🎤 Voice Cloning Setup")
    st.info("Upload audio samples of your voice to create a custom voice clone with ElevenLabs.")
    
    uploaded_files = st.file_uploader(
        "Upload voice samples (MP3, WAV, M4A)",
        type=['mp3', 'wav', 'm4a'],
        accept_multiple_files=True,
        help="Upload 1-25 audio files of your voice (each 1-30 minutes). Higher quality samples = better voice clone."
    )
    
    voice_name = st.text_input("Voice Clone Name", placeholder="My Custom Voice")
    voice_description = st.text_area("Voice Description", placeholder="Describe the voice characteristics...")
    
    if uploaded_files and voice_name and st.button("Create Voice Clone"):
        try:
            elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
            if not elevenlabs_api_key:
                st.error("ElevenLabs API key not found. Please add it to your secrets.")
                return None
                
            url = "https://api.elevenlabs.io/v1/voices/add"
            headers = {"xi-api-key": elevenlabs_api_key}
            
            files = []
            for uploaded_file in uploaded_files:
                files.append(('files', (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)))
            
            data = {
                'name': voice_name,
                'description': voice_description,
                'labels': '{"accent": "american", "age": "young", "gender": "male"}'  # Adjust as needed
            }
            
            # Debug: Show what we're sending
            st.write(f"Creating voice with {len(uploaded_files)} files...")
            st.write(f"Voice ID will be used: qpY4eaUGeCWHEUIVdbGM")
            
            with st.spinner("Creating voice clone... This may take a few minutes."):
                response = requests.post(url, headers=headers, files=files, data=data)
                
            if response.status_code == 200:
                result = response.json()
                voice_id = result.get("voice_id")
                st.success(f"✅ Voice clone '{voice_name}' created successfully!")
                st.info(f"Voice ID: {voice_id}")
                st.balloons()
                return voice_id
            else:
                st.error(f"Failed to create voice clone: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Error creating voice clone: {str(e)}")
            return None
    
    return None

def get_transcript(video_id):
    """Get transcript for a YouTube video."""
    import time
    import random
    
    try:
        # Add a small random delay to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        
        # Try to get transcript directly first
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
            return ' '.join([t['text'] for t in transcript_list])
        except Exception as e:
            # If direct method fails, try alternative approach
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                return ' '.join([t['text'] for t in transcript_list])
            except Exception as e2:
                st.error("No transcripts available for this video. Please try a different video or use the manual transcript option.")
                return None
        
    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            st.error("YouTube is rate limiting transcript requests. Please wait a few minutes and try again.")
        else:
            st.error(f"Error accessing transcripts: {str(e)}. Please use the manual transcript option.")
        return None

def process_transcript(transcript, chunk_size=1000, chunk_overlap=200):
    """Split transcript into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(transcript)

def create_vector_store(chunks):
    """Create a FAISS vector store from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    return FAISS.from_texts(chunks, embeddings)

def setup_qa_chain(vector_store):
    """Set up the QA chain with the vector store."""
    try:
        # Try OpenAI first
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        # Test the API key with a simple call
        test_response = llm.invoke("test")
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            st.warning("OpenAI quota exceeded. Using simple text-based fallback...")
            
            # Simple text-based QA function
            def simple_qa_function(query, context_docs):
                # Combine context documents
                context = "\n".join([doc.page_content for doc in context_docs])
                
                # Simple keyword-based matching
                query_lower = query.lower()
                context_lower = context.lower()
                
                # Find the most relevant sentences
                sentences = context.split('.')
                relevant_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 10:  # Skip very short sentences
                        # Check if query keywords appear in sentence
                        query_words = query_lower.split()
                        sentence_lower = sentence.lower()
                        
                        score = 0
                        for word in query_words:
                            if len(word) > 2 and word in sentence_lower:
                                score += 1
                        
                        if score > 0:
                            relevant_sentences.append((sentence, score))
                
                # Sort by relevance score
                relevant_sentences.sort(key=lambda x: x[1], reverse=True)
                
                if relevant_sentences:
                    # Return top 3 most relevant sentences
                    answer = ". ".join([sent[0] for sent in relevant_sentences[:3]])
                    return f"Based on the transcript: {answer}"
                else:
                    return "I couldn't find specific information about that in the transcript. Please try rephrasing your question or ask about different topics covered in the video."
            
            # Create a custom retrieval QA class
            class SimpleRetrievalQA:
                def __init__(self, retriever, qa_function):
                    self.retriever = retriever
                    self.qa_function = qa_function
                
                def run(self, query):
                    docs = self.retriever.invoke(query)
                    return self.qa_function(query, docs)
            
            return SimpleRetrievalQA(vector_store.as_retriever(), simple_qa_function)
        else:
            raise e

def main():
    # Load environment variables
    load_dotenv()
    
    st.title("YouTube Video Q&A with RAG")
    
    # Show API status
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("No OpenAI API key found. The app will use a free local model for question answering.")
    else:
        st.info("OpenAI API key detected. Will try OpenAI first, then fallback to local model if quota exceeded.")
    
    # Show ElevenLabs status and voice selection
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    if elevenlabs_api_key:
        st.success("🔊 Voice cloning enabled with ElevenLabs!")
        
        # Voice selection in sidebar
        with st.sidebar:
            st.subheader("🎤 Voice Settings")
            
            # Get available voices
            voices = get_elevenlabs_voices()
            if voices:
                voice_options = {name: voice_id for voice_id, name in voices}
                selected_voice_name = st.selectbox(
                    "Select Voice:",
                    options=list(voice_options.keys()),
                    help="Choose your custom voice or a preset voice"
                )
                selected_voice_id = voice_options.get(selected_voice_name)
                
                # Store selected voice in session state
                st.session_state.selected_voice_id = selected_voice_id
                st.info(f"Using voice: {selected_voice_name}")
            else:
                st.warning("No voices found. Create a voice clone below.")
                st.session_state.selected_voice_id = None
            
            # Voice cloning section
            st.markdown("---")
            upload_voice_sample()
    else:
        st.info("💡 Add ELEVENLABS_API_KEY to enable voice cloning with your own voice.")
    
    # Sidebar for video URL input
    with st.sidebar:
        st.header("YouTube Video")
        video_url = st.text_input("Enter YouTube Video URL:")
        
        # Manual transcript input option
        st.subheader("Alternative: Manual Transcript")
        manual_transcript = st.text_area("Or paste transcript text directly (if YouTube API is rate limited):", height=100)
        
        if manual_transcript:
            st.success("Using manual transcript!")
            if 'transcript' not in st.session_state or st.session_state.get('transcript') != manual_transcript:
                with st.spinner("Processing manual transcript..."):
                    st.session_state.transcript = manual_transcript
                    st.session_state.current_video_id = "manual"
                    chunks = process_transcript(manual_transcript)
                    st.session_state.vector_store = create_vector_store(chunks)
                    st.session_state.qa_chain = setup_qa_chain(st.session_state.vector_store)
                    st.success("Manual transcript processed successfully!")
        
        elif video_url:
            video_id = get_video_id(video_url)
            if video_id:
                # Debug: Show the extracted video ID
                st.caption(f"Video ID: {video_id}")
                
                # Always show the video player first
                st.video(video_url)
                
                # Try to get video metadata, but don't fail if it doesn't work
                try:
                    yt = YouTube(f"https://youtube.com/watch?v={video_id}")
                    st.subheader(yt.title)
                    st.caption(f"By: {yt.author}")
                except Exception as e:
                    st.subheader("YouTube Video")
                    st.caption("(Video metadata unavailable)")
                    # Only show warning if it's not a common HTTP error
                    if "400" not in str(e) and "403" not in str(e):
                        st.info("Note: Could not fetch video details, but transcript processing will still work.")
                
                if 'transcript' not in st.session_state or st.session_state.get('current_video_id') != video_id:
                    with st.spinner("Processing video transcript..."):
                        transcript = get_transcript(video_id)
                        if transcript:
                            st.session_state.transcript = transcript
                            st.session_state.current_video_id = video_id
                            chunks = process_transcript(transcript)
                            st.session_state.vector_store = create_vector_store(chunks)
                            st.session_state.qa_chain = setup_qa_chain(st.session_state.vector_store)
                            st.success("Video processed successfully!")
            else:
                st.error("Invalid YouTube URL - could not extract video ID")
    
    # Main chat interface
    if 'qa_chain' in st.session_state:
        st.subheader("Ask me anything about the video!")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the video:"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.qa_chain.run(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Add voice readout option with ElevenLabs
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if st.button("🔊 Listen", key=f"voice_{len(st.session_state.messages)}"):
                                with st.spinner("Generating voice with your custom voice..."):
                                    # Use selected voice or user's custom voice
                                    voice_id = getattr(st.session_state, 'selected_voice_id', "qpY4eaUGeCWHEUIVdbGM")
                                    audio_data = text_to_speech_elevenlabs(response, voice_id)
                                    if audio_data:
                                        # Convert audio data to base64 for HTML audio player
                                        audio_base64 = base64.b64encode(audio_data).decode()
                                        audio_html = f"""
                                        <audio controls autoplay>
                                            <source src="data:audio/mpeg;base64,{audio_base64}" type="audio/mpeg">
                                            Your browser does not support the audio element.
                                        </audio>
                                        """
                                        st.markdown(audio_html, unsafe_allow_html=True)
                                    else:
                                        st.warning("Voice generation failed. Please check your ElevenLabs API key and voice selection.")
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a YouTube video URL in the sidebar to get started.")

if __name__ == "__main__":
    main()

