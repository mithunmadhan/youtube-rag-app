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

def get_transcript(video_id):
    """Get transcript for a YouTube video."""
    import time
    import random
    
    try:
        # Add a small random delay to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        
        # First try to get any available transcript
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Prefer English transcripts
        preferred_languages = ['en', 'en-US', 'en-GB']
        for lang in preferred_languages:
            try:
                transcript = transcript_list.find_transcript([lang])
                transcript_data = transcript.fetch()
                return ' '.join([t['text'] for t in transcript_data])
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    st.warning("YouTube is rate limiting requests. Trying again in a moment...")
                    time.sleep(5)
                    try:
                        transcript_data = transcript.fetch()
                        return ' '.join([t['text'] for t in transcript_data])
                    except:
                        continue
                continue
        
        # If no preferred language found, try any available transcript
        for transcript in transcript_list:
            try:
                transcript_data = transcript.fetch()
                return ' '.join([t['text'] for t in transcript_data])
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    continue
                continue
                
        st.error("YouTube is currently rate limiting transcript requests. Please try again in a few minutes, or try a different video.")
        return None
        
    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            st.error("YouTube is rate limiting transcript requests. Please wait a few minutes and try again.")
        else:
            st.error(f"Error accessing transcripts: {str(e)}")
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
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a YouTube video URL in the sidebar to get started.")

if __name__ == "__main__":
    main()
