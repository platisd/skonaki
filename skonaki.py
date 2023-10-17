#!/usr/bin/env python3
"""
Summarize the supplied media file using OpenAI
"""

import argparse
import openai
import sys
import os
import mimetypes
import tempfile
import pysrt
import datetime

from pathlib import Path
from pydub import AudioSegment

TWENTYFIVE_MB = 26214400
TEMP_DIR = Path(tempfile.gettempdir())
DEFAULT_SUMMARY_PROMPT = (
    "Create a cheatsheet out of the following transcript in less than 50 words: \n"
)
CONTINUE_SUMMARY_PROMPT = "Continue with the next part of the same transcript, use the same style as before: \n"
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful assistant who summarizes with bullet points.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", help="Path to media file", type=Path)
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (default: read from OPENAI_API_KEY environment variable)",
        default=os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--transcription-prompt",
        help="Prompt to use for transcribing the video (e.g. What the video is about)",
        default="",
    )
    parser.add_argument(
        "--summary-prompt",
        help="Override the default prompt for summarizing the video",
        default=DEFAULT_SUMMARY_PROMPT,
    )
    parser.add_argument(
        "--frequency",
        help="How often (in sec) to create summaries of the video (default: 60)",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--model",
        help="OpenAI model to use (default: gpt-3.5-turbo)",
        default="gpt-3.5-turbo",
    )
    parser.add_argument(
        "--language",
        help="Language of the input media for transcribing"
        + " (default: en, must be in ISO 639-1 format and supported by OpenAI's Whisper API)."
        + " For translating, the language is automatically detected"
        + " and the output language is always English.",
    )
    parser.add_argument(
        "--output",
        help="Output file (default: only stdout)",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    exit_code, exit_message = generate_summary(
        media=args.media,
        api_key=args.api_key,
        transcription_prompt=args.transcription_prompt,
        summary_prompt=args.summary_prompt,
        model=args.model,
        language=args.language,
        frequency=args.frequency,
        output=args.output,
    )
    print(exit_message)
    return exit_code


def generate_summary(
    media: Path,
    api_key: str = os.environ.get("OPENAI_API_KEY"),
    transcription_prompt: str = "",
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
    model: str = "gpt-3.5-turbo",
    language: str = "en",
    frequency: int = 60,
    output: Path = None,
):
    if not media.is_file():
        exit_message = f"Media file {media} does not exist"
        return (1, exit_message)

    if not api_key:
        exit_message = (
            "OpenAI API key is required, none provided or found in environment"
        )
        return (1, exit_message)

    audio = get_audio(media)
    audio_size = audio.stat().st_size
    if audio_size > TWENTYFIVE_MB:
        print(
            f"Audio file is too large {audio_size / 1000000}MB, must be less than 25MB, attempting to downsample"
        )
        audio = downsample_audio(audio, TWENTYFIVE_MB)
        audio_size = audio.stat().st_size
    print(f"Audio file size in MB: {audio_size / 1000000}")

    openai.api_key = api_key
    print(f"Transcribing using OpenAI's Whisper AI")
    with open(audio, "rb") as f:
        transcript = openai.Audio.transcribe(
            "whisper-1",
            f,
            response_format="srt",
            language=language,
            prompt=transcription_prompt,
        )

    subs = pysrt.from_string(transcript)
    # Break the transcript into chunks based on the frequency
    chunks = []
    chunk = []
    chunk_beginning = subs[0].start.ordinal
    for sub in subs:
        chunk.append(sub)
        if sub.start.ordinal - chunk_beginning > frequency * 1000:
            chunks.append((chunk, chunk_beginning))
            chunk = []
            chunk_beginning = sub.start.ordinal
    if chunk:
        chunks.append((chunk, chunk_beginning))

    messages = [SYSTEM_PROMPT]
    cheatsheet = ""
    current_chunk = 1
    for subtitle_chunk, chunk_timestamp in chunks:
        # Convert the chunk to text
        text = "\n".join([sub.text for sub in subtitle_chunk])

        # Count the number of characters in messages
        characters_per_token = 4
        max_tokens = get_max_tokens(model)
        if get_characters(messages) > max_tokens * characters_per_token:
            # Keep only the first message (system prompt) and the last message (assistant response)
            print("Reached the max number of tokens, resetting messages")
            assert len(messages) > 2
            messages = [messages[0], messages[-1]]
            # There's a chance that the assistant response is too long, so trim
            if get_characters(messages) > max_tokens * characters_per_token:
                print("The last message is too long, trim it to the max length")
                messages[-1]["content"] = messages[-1]["content"][
                    max_tokens * characters_per_token :
                ]
                messages[-1]["content"] = "..." + messages[-1]["content"]

        continue_or_first_prompt = (
            CONTINUE_SUMMARY_PROMPT if len(messages) > 1 else summary_prompt
        )
        summary_prompt = continue_or_first_prompt + "\n" + text
        messages.append(
            {
                "role": "user",
                "content": text,
            },
        )

        print(
            f"Summarizing using OpenAI's {model} model. Part {current_chunk} of {len(chunks)}."
        )
        current_chunk += 1
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.6,
        )
        gpt_response = response.choices[0].message.content
        # Format timestamp in hh:mm:ss format
        chunk_timedelta = datetime.timedelta(milliseconds=chunk_timestamp)
        chunk_timedelta_str = str(chunk_timedelta).split(".")[0]
        # If hours is only 1 digit, add a leading 0
        if len(chunk_timedelta_str.split(":")[0]) == 1:
            chunk_timedelta_str = "0" + chunk_timedelta_str

        cheatsheet += f"{chunk_timedelta_str}\n{gpt_response}\n"
        messages.append(
            {
                "role": "assistant",
                "content": gpt_response,
            },
        )

    if output:
        output.write_text(cheatsheet)
        print(f"Saved cheatsheet to {output.resolve()}")

    exit_message = "\n\n\n" + cheatsheet
    return (0, exit_message)


def get_characters(messages: list):
    return sum([len(message["content"]) for message in messages])


def get_max_tokens(model: str):
    if model == "gpt-4":
        return 7000
    else:
        return 3000


def get_audio(media: Path):
    print(f"Getting audio from {media}")
    type = mimetypes.guess_type(media)[0]
    if type == "audio":
        print("Media is already audio, no need to convert")
        return media

    audio = TEMP_DIR / "audio.mp3"
    AudioSegment.from_file(media).set_channels(1).export(
        audio, format="mp3", bitrate="128k"
    )
    print(f"Split audio file and saved to {audio}")
    return audio


def downsample_audio(audio: Path, max_size: int = TWENTYFIVE_MB):
    print(f"Downsampling audio from {audio}")
    bitrates = ["64k", "32k", "16k"]
    for bitrate in bitrates:
        downsampled = TEMP_DIR / "audio_downsampled.mp3"
        AudioSegment.from_file(audio).set_channels(1).export(
            downsampled, format="mp3", bitrate=bitrate
        )
        if downsampled.stat().st_size < max_size:
            print(
                f"Downsampled audio file and saved to {downsampled} with bitrate {bitrate}"
            )
            return downsampled

    print("Unable to downsample audio file, it needs to be split into smaller chunks")
    print("Open a feature request on GitHub if you need this feature")
    raise Exception("Unable to downsample audio file")


if __name__ == "__main__":
    sys.exit(main())
