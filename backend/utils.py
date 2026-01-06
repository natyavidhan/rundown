import yt_dlp
import os
import whisper

def download_audio(url: str, output_dir: str = "downloads") -> str:
    """
    Downloads a YouTube video as an MP3 file.
    
    Args:
        url: The YouTube video URL.
        output_dir: The directory to save the MP3 file.
        
    Returns:
        The path to the downloaded MP3 file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(filename)
        final_path = f"{base}.mp3"
        
    return final_path

def transcribe_audio(audio_path: str, model_name: str = "base") -> list:
    """
    Transcribes an audio file using OpenAI's Whisper model and returns a list of segments with timeline.
    
    Args:
        audio_path: Path to the audio file (e.g., MP3).
        model_name: The name of the Whisper model to load (default: "base").
        
    Returns:
        A list of dictionaries containing 'start', 'end', and 'text' for each segment.
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    
    transcript = []
    for segment in result["segments"]:
        transcript.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip()
        })
        
    return transcript

from groq import Groq
import json

def generate_video_gist(transcript: list, model: str = "llama3-70b-8192") -> list:
    """
    Uses Groq LLM to analyze the transcript and select key segments for a concise summary.
    
    Args:
        transcript: List of segments with 'start', 'end', and 'text'.
        model: Groq model to use.
        
    Returns:
        A list of selected segments with 'start', 'end', and 'summary'.
    """
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # Prepare transcript text for the prompt
    transcript_text = "\n".join([f"[{t['start']:.2f}-{t['end']:.2f}] {t['text']}" for t in transcript])
    
    prompt = f"""
    Here is a transcript of a video with timestamps:
    
    {transcript_text}
    
    Your task is to identify the most critical segments that a viewer should watch to get the complete gist of the video without watching the entire thing. 
    Select a series of continuous time ranges. Merge adjacent or close relevant segments if needed.
    
    Return ONLY a valid JSON object with a single key 'highlights' which is a list of objects. Each object must have:
    - 'start': start time in seconds (float)
    - 'end': end time in seconds (float)
    - 'summary': a brief description of what is covered in this segment
    
    Do not include any other text or markdown formatting.
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes video transcripts to create concise summaries."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        response_format={"type": "json_object"},
    )
    
    response_content = chat_completion.choices[0].message.content
    try:
        data = json.loads(response_content)
        return data.get("highlights", [])
    except json.JSONDecodeError:
        print("Failed to decode JSON from LLM response")
        return []
