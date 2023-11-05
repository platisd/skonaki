# skonaki ✍️

Study smarter, not harder: Create cheatsheets out of videos 🪄

![skonaki logo](skonaki-logo-small.png)


## What is `skonaki`?

`skonaki` is a tool that allows you to create cheatsheets out of videos.<br>
It is a command line tool that takes a path (or a URL) to a video or audio file and produces
a list of bullet points with the most important information from the video.
The bullet points are accompanied by timestamps, in case you need to refer back
to the video and check a specific part for yourself.<br>
You can determine how often you would like to "sample" the video and, if you choose,
you could take control of the cheatsheet generation process by providing your own
prompts.

While `skonaki` cannot and should not replace studying, it can accelerate your learning
process by allowing you to quickly go through recordings and produce a set of notes
that you can further refine and study from.

The name `skonaki` comes from the Greek word "σκονάκι" (skonaki), which means "cheatsheet".

![skonaki sample output](skonaki-output.png)

## Why use `skonaki`?

`skonaki` is a tool that can help you study smarter, not harder.
I used it to avoid having to watch the video lectures at a course I am taking.
You could use it to create notes out of your recordings, meetings, or even podcasts.

## How does `skonaki` work?

Ultimately, what `skonaki` does is split audio from video, then transcribe it using either
the Whisper OpenAI API or by running Whisper locally and finally use the transcription to create a cheatsheet with the most
important information from the video.
The cheatsheet generation is done by using the ChatCompletion API by OpenAI.
It's been tested to work on Ubuntu with `ffmpeg` installed.<br>
If a URL is provided, `skonaki` will download the video using [yt-dlp](https://github.com/yt-dlp/yt-dlp).
Currently, only publicly available media are supported. Create an issue if you need support for
something more or download the video yourself and pass the path to `skonaki`.

### Installation

* Dependencies
  * To use `skonaki` you need to install its dependencies first by `pip install --user -r requirements.txt`.
  * The Whisper OpenAI API is used by default. To instead use Whisper locally, install the Whisper library (including all other dependencies) by `pip install --user -r requirements-local-whisper.txt`.
* OpenAI API key
  * You need to get an [OpenAI API key](https://platform.openai.com/account/api-keys).
    This is a paid service and it's used for the transcription and the cheatsheet generation.
  * Set the `OPENAI_API_KEY` environment variable to your API key or pass it as an argument to `skonaki.py`.

Run `skonaki.py --help` to see all available options.

### Comparison with other tools

Several tools help create notes, summaries, etc. out of videos.
The main difference is that most of them are web services that only require you to
provide a link to the video and they do the rest for you.
I wanted to avoid handling video URLs since that's its own project and didn't want to bother.

Here's my very "scientific" comparison of `skonaki` with other tools.
I did not bother with those that required signing up or had a paywall:

* [youtubesummarizer.com](https://youtubesummarizer.com/)
  * Works pretty well, there are some ads and I like the executive summary part that they also provide. I'd say the quality of the results is similar to `skonaki`.
* [summarize.tech](https://www.summarize.tech/)
  * Wasn't very impressed with the results. It summarizes OKish but wouldn't be good enough for studying purposes.
* [notegpt.io](https://notegpt.io/)
  * Rather quick, I only tried their web version, while it also comes as a plugin. While the summary was on point, I found it too short to use for studying purposes. Great for summarizing though.

| Tool                  | Summaries | Timestamps | Customization | Notes                                                                                      |
| --------------------- | --------- | ---------- | ------------- | ------------------------------------------------------------------------------------------ |
| `skonaki`             | 🏆         | ✅          | ✅             | Can customize anything from the prompt used, to how often a "summary" is produced          |
| youtubesummarizer.com | 🏆         | ❌          | 😒             | There's a sliding scale for how long the summary should be, but doesn't get too "detailed" |
| summarize.tech        | 😒         | ✅          | ❌             | No customization, their summary is not very suitable for studying                          |
| notegpt.io            | 😒         | ❌          | ❌             | The summary is pretty good, but not detailed enough for studying                           |
