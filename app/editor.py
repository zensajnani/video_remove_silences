from dotenv import load_dotenv
import os
import ffmpeg
import json
import anthropic
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize AI clients
anthropic_client = anthropic.Anthropic()
deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

# Define system prompt
SYSTEM_PROMPT = '''You are an expert video editor tasked with creating a concise, coherent video from a transcript. You will receive an array of objects, each containing a word, its start time, and end time in milliseconds. Your goal is to create a JSON object specifying which words to include in the final edit.

Editing guidelines:
1. Use only words provided in the input array.
2. Maintain logical flow and context in the edited video.
3. Remove repeated sentences, keeping the latter instance if it seems more polished.
4. Eliminate false starts and filler words that don't contribute to the message.
5. Cut out extended silences (>1 second) between words or sentences.
6. Ensure sentences are complete and not cut off mid-thought.
7. Add a small buffer (50-100ms) at the end of sentences for natural pacing.
8. Aim for a concise video while preserving the core message and context.
9. Ensure no single pause between words exceeds 500ms unless it's a natural break point.
10. Check that you have not included false starts and only sentences that are finished properly by the speaker.
11. Return JSON and only JSON.
12. Cut out any 'um's, 'ah's, or other filler words.

Example output:
{
    "desired_transcription": "hello this is a test video",
    "transcription_sources": [
        {"file": "video.mp4", "start": 0.24, "end": 2.82}
    ]
}'''

def process_video(video_files):
    # Temporary files to store the MP3s
    temp_mp3_files = []
    simplified_words = []

    try:
        # Convert videos to MP3 and transcribe
        for idx, video_file in enumerate(video_files):
            temp_file = f'temp_audio_{idx}.mp3'
            temp_mp3_files.append(temp_file)
            
            # Convert to MP3
            (ffmpeg
             .input(video_file)
             .output(temp_file, format='mp3')
             .global_args('-loglevel', 'error')
             .overwrite_output()
             .run())

            # Get transcription from Deepgram
            with open(temp_file, "rb") as audio_file:
                buffer_data = audio_file.read()

            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                filler_words=True
            )

            response = deepgram.listen.rest.v("1").transcribe_file(
                {"buffer": buffer_data}, options
            )

            # Process words
            file_string = f"{video_file}: "
            simplified_words.append(file_string)
            
            for word in response.results.channels[0].alternatives[0].words:
                simplified_words.append({
                    "word": word.word,
                    "start": float(word.start),
                    "end": float(word.end),
                    "file": video_file
                })

        # Get editing instructions from Claude
        text_completion = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": str(simplified_words)}]
        )
        
        response = text_completion.content[0].text
        edited_transcript_json = json.loads(response)
        
        # Create output video
        editing_instructions = edited_transcript_json['transcription_sources']
        inputs = [
            ffmpeg.input(cut['file'], ss=cut['start'], to=cut['end']) 
            for cut in editing_instructions
        ]
        
        stream_pairs = [(input.video, input.audio) for input in inputs]
        
        if stream_pairs:
            concat = ffmpeg.concat(
                *[item for sublist in stream_pairs for item in sublist], 
                v=1, a=1
            ).node
            output = ffmpeg.output(concat[0], concat[1], 'ai_output.mp4')
            output.run()

        return edited_transcript_json

    finally:
        # Cleanup temp files
        for temp_file in temp_mp3_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    video_files = ['test_clip_trimmed.mp4']
    
    print("\nProcessing video...")
    result = process_video(video_files)
    
    print("\nEdited script:")
    print(result['desired_transcription'])
    
    print("\nEditing instructions:")
    for cut in result['transcription_sources']:
        print(f"File: {cut['file']}")
        print(f"Time: {cut['start']:.2f}s - {cut['end']:.2f}s")
        print()
    
    print("Video saved as 'ai_output.mp4'")
