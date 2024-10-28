import os
import ffmpeg
import json
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import anthropic
from typing import List
import tempfile
import logging
from dotenv import load_dotenv
from datetime import timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize AI clients
anthropic_client = anthropic.Anthropic()
speech_client = SpeechClient()

# Update the SYSTEM_PROMPT to specify exact JSON format we want
SYSTEM_PROMPT = '''You are an expert video editor tasked with creating a concise, coherent video from a transcript. You will receive an array of objects, each containing a word, its start time, and end time in seconds. Your goal is to create a JSON object specifying which words to include in the final edit.

Return a JSON object with exactly this structure:
{
    "transcription_sources": [
        {
            "word": "word_text",
            "start": start_time_in_seconds,
            "end": end_time_in_seconds,
            "file": "filename"
        }
    ],
    "desired_transcription": "full edited transcript as text"
}

Editing guidelines:
1. Use only words provided in the input array with their exact timestamps.
2. Maintain logical flow and context in the edited video.
3. Remove repeated sentences, keeping the latter instance if it seems more polished.
4. Eliminate false starts and filler words that don't contribute to the message.
5. Cut out extended silences (>1 second) between words or sentences.
6. Ensure sentences are complete and not cut off mid-thought.
7. Add a small buffer (50-100ms) at the end of sentences for natural pacing.
8. Aim for a concise video while preserving the core message and context.
9. Ensure no single pause between words exceeds 500ms unless it's a natural break point. 
10. Check that you have not included false starts and only sentences that are finished properly by the speaker.
11. Be aggressive in removing filler words like "um", "uh", "like", "you know", etc.
12. Return JSON exactly in the format specified above.'''

def parse_time_offset(offset) -> float:
    """Convert Google Cloud time offset to seconds"""
    if isinstance(offset, timedelta):
        # If it's already a timedelta, just convert to seconds
        return offset.total_seconds()
    elif isinstance(offset, str):
        # If it's a string, parse it
        parts = offset.strip('"').split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Unexpected offset type: {type(offset)}")

async def process_videos(video_files: List[str]) -> dict:
    temp_files = []
    simplified_words = []
    original_transcripts = []

    try:
        # Convert videos and transcribe
        for idx, video_file in enumerate(video_files):
            # First convert video to WAV with correct format for Google Speech
            temp_wav = f'temp_audio_{idx}.wav'
            temp_files.append(temp_wav)
            
            try:
                # Convert to WAV using exact command that works
                (ffmpeg
                 .input(video_file)
                 .output(temp_wav,
                        ar=16000,      # Sample rate: 16kHz
                        ac=1,          # Audio channels: 1 (mono)
                        acodec='pcm_s16le'  # Codec: 16-bit PCM
                 )
                 .global_args('-loglevel', 'error')
                 .overwrite_output()
                 .run())

                # Configure recognition with explicit US English
                config = cloud_speech.RecognitionConfig(
                    explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                        encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=16000,
                        audio_channel_count=1
                    ),
                    features=cloud_speech.RecognitionFeatures(
                        enable_word_confidence=True,
                        enable_word_time_offsets=True,
                    ),
                    model="long",
                    language_codes=["en-US"],
                )

                # Create the request
                with open(temp_wav, "rb") as audio_file:
                    content = audio_file.read()

                request = cloud_speech.RecognizeRequest(
                    recognizer=f"projects/{os.getenv('GOOGLE_CLOUD_PROJECT')}/locations/global/recognizers/_",
                    config=config,
                    content=content
                )

                # Get transcription
                response = speech_client.recognize(request=request)

                # Process results
                for result in response.results:
                    for alternative in result.alternatives:
                        original_transcripts.append(alternative.transcript)
                        for word in alternative.words:
                            simplified_words.append({
                                "word": word.word,
                                "start": parse_time_offset(word.start_offset),
                                "end": parse_time_offset(word.end_offset),
                                "file": video_file  # Keep original video file for final edit
                            })

            except Exception as e:
                logger.error(f"Error processing video {video_file}: {str(e)}")
                raise

        # Process with Anthropic
        try:
            text_completion = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user", 
                    "content": f"""Here are the words with their timestamps from the video(s): {json.dumps(simplified_words)}

Please create a highly condensed version by removing all silences, false starts, and filler words.
Focus on maintaining only the essential content while removing all unnecessary elements.
Return a JSON object with the exact structure specified in the system prompt, using the original timestamps from the input."""
                }]
            )

            response = text_completion.content[0].text
            logger.info(f"Raw Claude response: {response}")
            
            edited_transcript_json = json.loads(response)

            # Generate output video
            editing_instructions = edited_transcript_json['transcription_sources']
            
            # Create video segments with both video and audio
            inputs = [ffmpeg.input(cut['file'], ss=cut['start'], t=cut['end']-cut['start']) 
                     for cut in editing_instructions]
            
            output_file = "ai_output.mp4"
            
            if inputs:
                # Concatenate video segments
                stream_pairs = [(input.video, input.audio) for input in inputs]
                concat_streams = ffmpeg.concat(*[item for sublist in stream_pairs for item in sublist], 
                                            v=1, a=1)
                
                # Create a complex filter for audio silence removal
                # Keep video stream as is, only process audio
                output = (
                    ffmpeg
                    .output(
                        # Video stream goes directly to output
                        concat_streams['v'],
                        # Audio stream goes through silence removal
                        concat_streams['a'].filter('silenceremove',
                            stop_periods='-1',
                            stop_duration='1',
                            stop_threshold='-60dB'
                        ),
                        output_file,
                        acodec='aac',
                        vcodec='copy'
                    )
                    .global_args('-loglevel', 'error')
                    .overwrite_output()
                )
                
                # Run the ffmpeg command
                output.run()

            return {
                "original_transcripts": original_transcripts,
                "edited_script": edited_transcript_json['desired_transcription'],
                "output_file": output_file,
                "word_timestamps": edited_transcript_json['transcription_sources']
            }

        except Exception as e:
            logger.error(f"Error with AI processing: {str(e)}")
            raise

    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    try:
        import asyncio
        
        video_file = "silence-remover-test-vid.mp4"  # Changed to .mp4
        result = asyncio.run(process_videos([video_file]))
        
        print("\nProcessing Results:")
        print("Original Transcript:")
        print(result["original_transcripts"][0])
        print("\nEdited Transcript:")
        print(result["edited_script"])
        print("\nWord Timestamps:")
        for word in result["word_timestamps"]:
            print(f"{word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")
        print(f"\nOutput saved to: {result['output_file']}")
        
    except Exception as e:
        logger.error(f"Failed to run editor: {e}")
