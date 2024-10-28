import os
import ffmpeg
import json
from openai import OpenAI
import anthropic
from typing import List
import tempfile
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize AI clients
openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()

SYSTEM_PROMPT = '''You are an expert video editor tasked with creating a concise, coherent video from a transcript. You will receive an array of objects, each containing a word, its start time, and end time in milliseconds. Your goal is to create a JSON object specifying which words to include in the final edit.

Editing guidelines:
1. Use only words provided in the input array.
2. Maintain logical flow and context in the edited video.
3. Remove repeated sentences, keeping the latter instance if it seems more polished.
4. Eliminate false starts and filler words that don't contribute to the message. For example if in the transcript you see the words "I... I am.... I am going to" remove the first two false starts and just include the timestamps from the beginning of the proper full sentence.
5. Cut out extended silences (>1 second) between words or sentences.
6. Ensure sentences are complete and not cut off mid-thought.
7. Add a small buffer (50-100ms) at the end of sentences for natural pacing.
8. Aim for a concise video while preserving the core message and context.
9. Ensure no single pause between words exceeds 500ms unless it's a natural break point. 
10. Check that you have not included false starts and only sentences that are finished properly by the speaker.
11. Be aggressive in removing filler words like "um", "uh", "like", "you know", etc.
12. Return JSON and only JSON.'''

async def process_videos(video_files: List[str]) -> dict:
    temp_mp3_files = []
    transcriptions = []
    simplified_words = []
    original_transcripts = []

    try:
        # Convert videos to MP3 and transcribe
        for idx, video_file in enumerate(video_files):
            temp_file = f'temp_audio_{idx}.mp3'
            temp_mp3_files.append(temp_file)
            
            try:
                # Suppress ffmpeg output
                (ffmpeg
                 .input(video_file)
                 .output(temp_file, format='mp3')
                 .global_args('-loglevel', 'quiet')
                 .run())

                with open(temp_file, "rb") as audio_file:
                    transcript = openai_client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-1",
                        response_format="verbose_json",
                        timestamp_granularities=["word"]
                    )

                # Store original transcript
                original_transcripts.append(transcript.text)
                
                # Format word data
                file_string = f"{video_file}: "
                simplified_words.append(file_string)
                
                # Debug print to see the structure
                logger.info(f"First word structure: {transcript.words[0]}")
                
                # Fix the word access - using word directly instead of attributes
                for word in transcript.words:
                    simplified_words.append({
                        "word": word.word,  # Changed from text to word
                        "start": float(word.start),  # Ensure it's a float
                        "end": float(word.end),  # Ensure it's a float
                        "file": video_file
                    })

            except Exception as e:
                logger.error(f"Error processing video {video_file}: {str(e)}")
                raise

        # Process with Anthropic
        prompt = f"""Here are the words with their timestamps from the video(s): {json.dumps(simplified_words)}

Please create a highly condensed version by removing all silences, false starts, and filler words.
Focus on maintaining only the essential content while removing all unnecessary elements.
Return ONLY a JSON object with the edited segments and transcript."""
        
        try:
            text_completion = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            response = text_completion.content[0].text
            logger.info(f"Raw Claude response: {response}")
            
            try:
                edited_transcript_json = json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error. Response was: {response}")
                logger.error(f"Error details: {str(e)}")
                raise

            # Log original and edited transcripts
            logger.info("Original Transcripts:")
            for idx, transcript in enumerate(original_transcripts):
                logger.info(f"Video {idx + 1}: {transcript}")
            
            logger.info("\nEdited Transcript:")
            logger.info(edited_transcript_json['desired_transcription'])

            # Generate output video
            editing_instructions = edited_transcript_json['transcription_sources']
            inputs = [ffmpeg.input(cut['file'], ss=cut['start'], t=cut['end']-cut['start']) 
                     for cut in editing_instructions]
            
            stream_pairs = [(input.video, input.audio) for input in inputs]

            output_file = "ai_output.mp4"
            if stream_pairs:
                concat = ffmpeg.concat(
                    *[item for sublist in stream_pairs for item in sublist], 
                    v=1, a=1
                ).node
                output = ffmpeg.output(concat[0], concat[1], output_file)
                output.run()

            return {
                "original_transcripts": original_transcripts,
                "edited_script": edited_transcript_json['desired_transcription'],
                "output_file": output_file
            }

        except Exception as e:
            logger.error(f"Error with AI processing: {str(e)}")
            raise

    finally:
        # Cleanup temp files
        for temp_file in temp_mp3_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def process_transcript(transcript: List[dict]) -> dict:
    """Process a transcript using AI to create editing instructions"""
    try:
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user", 
                "content": f"Here is the transcript to edit: {json.dumps(transcript)}"
            }]
        )
        return json.loads(response.content[0].text)
    except Exception as e:
        logger.error(f"Error processing transcript: {e}")
        raise

if __name__ == "__main__":
    # Test with actual audio file
    try:
        import asyncio
        
        video_file = "silence-remover-test-vid.mp3"
        result = asyncio.run(process_videos([video_file]))
        
        print("\nProcessing Results:")
        print("Original Transcript:")
        print(result["original_transcripts"][0])
        print("\nEdited Transcript:")
        print(result["edited_script"])
        print(f"\nOutput saved to: {result['output_file']}")
        
    except Exception as e:
        logger.error(f"Failed to run editor: {e}")
