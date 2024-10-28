import os
import ffmpeg
import json
import anthropic
from typing import List
import logging
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize AI clients
anthropic_client = anthropic.Anthropic()
deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

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

async def process_videos(video_files: List[str]) -> dict:
    temp_files = []
    simplified_words = []
    original_transcripts = []

    try:
        # Convert videos and transcribe
        for idx, video_file in enumerate(video_files):
            # First convert video to WAV
            temp_wav = f'temp_audio_{idx}.wav'
            temp_files.append(temp_wav)
            
            try:
                # Convert to WAV
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

                # Read the audio file
                with open(temp_wav, "rb") as file:
                    buffer_data = file.read()

                payload: FileSource = {
                    "buffer": buffer_data,
                }

                # Configure Deepgram options
                options = PrerecordedOptions(
                    model="nova-2",
                    smart_format=True,
                    diarize=True,  # Enable speaker detection
                    filler_words=True  # Explicitly enable filler word detection
                )

                # Get transcription using REST API
                response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
                
                # Process results
                for channel in response.results.channels:
                    for alternative in channel.alternatives:
                        original_transcripts.append(alternative.transcript)
                        for word in alternative.words:
                            word_data = {
                                "word": word.word,
                                "start": float(word.start),
                                "end": float(word.end),
                                "file": video_file
                            }
                            
                            # Add filler word detection if available
                            try:
                                if hasattr(word, "is_filler"):
                                    word_data["is_filler"] = word.is_filler
                            except:
                                word_data["is_filler"] = False
                                
                            simplified_words.append(word_data)

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
            
            # Create video segments with exact timestamps
            inputs = [
                ffmpeg.input(cut['file'], ss=cut['start'], t=cut['end']-cut['start']) 
                for cut in editing_instructions
            ]
            
            output_file = "ai_output.mp4"
            
            if inputs:
                # Simple concatenation
                stream_pairs = [(input.video, input.audio) for input in inputs]
                v1 = ffmpeg.concat(*[pair[0] for pair in stream_pairs], v=1, a=0)
                a1 = ffmpeg.concat(*[pair[1] for pair in stream_pairs], v=0, a=1)
                
                # Write final output without stream copy
                output = (
                    ffmpeg
                    .output(v1, a1, output_file,
                           acodec='aac')  # Only specify audio codec
                    .global_args('-loglevel', 'error')
                    .overwrite_output()
                )
                output.run()

            # After generating the output file
            if os.path.exists(output_file):
                # Get actual output duration using ffprobe
                probe = ffmpeg.probe(output_file)
                actual_duration = float(probe['format']['duration'])
                print(f"\nActual Output Duration: {actual_duration:.3f} seconds")

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
        
        video_file = "silence-remover-test-vid.mp4"
        
        print("\n" + "="*50)
        print("Starting Video Processing")
        print("="*50)
        
        result = asyncio.run(process_videos([video_file]))
        
        print("\nDeepgram Raw Transcription (with filler words):")
        print("-"*50)
        for idx, transcript in enumerate(result["original_transcripts"]):
            print(f"Alternative {idx + 1}:")
            print(transcript)
            print()
        
        print("\nDetailed Word Analysis:")
        print("-"*50)
        original_words = []
        for word in result["word_timestamps"]:
            start = f"{word['start']:.3f}".rjust(7)  # Changed to 3 decimal places
            end = f"{word['end']:.3f}".rjust(7)
            is_filler = word.get('is_filler', False)
            filler_mark = "[FILLER]" if is_filler else ""
            word_info = f"{start}s - {end}s: {word['word']:<15} {filler_mark}"
            original_words.append(word_info)
            
        # Print in columns for better readability
        col_width = max(len(word) for word in original_words) + 2
        num_cols = 2
        for i in range(0, len(original_words), num_cols):
            row_words = original_words[i:i + num_cols]
            print("".join(word.ljust(col_width) for word in row_words))
        
        print("\nEdited Version:")
        print("-"*50)
        print("Final Transcript:")
        print(result["edited_script"])
        print()
        
        print("Edited Segments:")
        for word in result["word_timestamps"]:
            start = f"{word['start']:.3f}".rjust(7)
            end = f"{word['end']:.3f}".rjust(7)
            duration = f"{(word['end'] - word['start']):.3f}".rjust(7)
            print(f"{start}s - {end}s ({duration}s): {word['word']}")
            
        # Calculate and show timing statistics
        if result["word_timestamps"]:
            first_word = result["word_timestamps"][0]
            last_word = result["word_timestamps"][-1]
            total_duration = last_word['end'] - first_word['start']
            
            # Calculate total speaking time (sum of word durations)
            speaking_time = sum(word['end'] - word['start'] for word in result["word_timestamps"])
            
            print(f"\nTiming Analysis:")
            print(f"Total Duration: {total_duration:.3f} seconds")
            print(f"Speaking Time: {speaking_time:.3f} seconds")
            print(f"Silence Removed: {(total_duration - speaking_time):.3f} seconds")
            print(f"Compression Ratio: {speaking_time/total_duration:.1%}")
        
        print(f"\nOutput saved to: {result['output_file']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Failed to run editor: {e}")
