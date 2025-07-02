import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
    get_job_context,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai, silero, noise_cancellation

# for audio support
from livekit import rtc

# for image support
from livekit.agents.llm import ImageContent
from livekit.agents.utils.images import encode, EncodeOptions, ResizeOptions

logger = logging.getLogger("experimance")

load_dotenv(override=True, dotenv_path=".env.local")

# load common instructions from a markdown file
# Determine the path to prompt_simple.md relative to this file
script_dir = Path(__file__).parent.parent.parent  # Go up to agent directory
prompt_path = script_dir / "prompts" / "prompt_simple.md"
with open(prompt_path, "r") as f:
    common_instructions = f.read()


@dataclass
class AudienceData:
    # This structure is passed as a parameter to function calls.
    name: Optional[str] = None
    location: Optional[str] = None

# Inspiration: https://github.com/livekit-examples/python-agents-examples/blob/main/complex-agents/personal_shopper/personal_shopper.py
@dataclass
class UserData:
    """Class to store user data and agents during a call."""
    personas: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None
    ctx: Optional[JobContext] = None

    # Customer information
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    location: Optional[str] = None

    def is_identified(self) -> bool:
        """Check if the customer is identified."""
        return self.first_name is not None

    def reset(self) -> None:
        """Reset customer information."""
        self.first_name = None
        self.last_name = None
        self.location = None

    def summarize(self) -> str:
        """Return a summary of the user data."""
        if self.is_identified():
            userinfo =  f"The user's name is {self.first_name}"
            if self.last_name:
                userinfo += f" {self.last_name}"
            if self.location:
                userinfo += f" from {self.location}"
            return userinfo
        return "User not yet identified."


class IntroAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"{common_instructions} Your goal is to introduce yourself and "
            "let the audience know they can interact with the art work by talking to you and playing with the sand."
            "Ask the person their name and where they are from, then immediately call `information_gathered`. " \
            "Do not ask any other questions."
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply()

    @function_tool
    async def information_gathered(
        self,
        context: RunContext[UserData],
        name: str,
        location: str | None = None,
    ):
        """Call after the user has provided some basic the information about themselves. 
        This function is called by the agent when the user provides at least their name 
        and optionally a location.

        Args:
            name: The name of the user
            location: The location of the user [optional]
        """
        # parse name to see if fist, or full
        if name and len(name.split(" ")) > 1:
            # if the name contains a space, assume it's a full name
            # and split it into first and last name
            first_name, last_name = name.split(" ", 1)
            context.userdata.first_name = first_name
            context.userdata.last_name = last_name
        else:
            # otherwise, assume it's just a first name
            context.userdata.first_name = name
        # if the user provided a location, store it
        context.userdata.location = location if (location and location != "") else None

        agent = ExperimanceAgent(context.userdata)
        # by default, agent will start with a new context, to carry through the current
        # chat history, pass in the chat_ctx
        # agent = ExperimanceAgent(name, location, chat_ctx=context.chat_ctx)

        logger.info(
            "switching to the main agent with the provided user data: %s", context.userdata
        )
        return agent


class ExperimanceAgent(Agent):
    def __init__(self, userdata: UserData, *, chat_ctx: Optional[ChatContext] = None) -> None:
        self._latest_frame = None  # latest video frame of the "vision" agent

        super().__init__(
            instructions=f"{common_instructions}. "
                f"{userdata.summarize()}"
                "You can control the images generated to an extent, "
                "you can choose a biome and/or location if the user requests it or you connect it to the conversation."
                "Biomes: forest, desert, tundra, island, tropical, jungle, prairie, mountains",
            # each agent could override any of the model services, including mixing
            # realtime and non-realtime models
            #llm=openai.realtime.RealtimeModel(voice="echo"),
            #tts=None,
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        # when the agent is added to the session, we'll initiate the conversation by
        # using the LLM to generate a reply
        self.session.generate_reply()

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        # Add the latest video frame, if any, to the new message
        if self._latest_frame:
            image_bytes = encode(
                    self._latest_frame,
                    EncodeOptions(
                        format="JPG",
                        resize_options=ResizeOptions(
                            width=512, 
                            height=512, 
                            strategy="scale_aspect_fit"
                        )
                    )
                )
            image_content = ImageContent(
                image=f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
            )
            new_message.content.append(image_content)
            self._latest_frame = None

    @function_tool
    async def interaction_finished(self, context: RunContext[UserData]):
        """When the user says goodbye or leaves the installation, call this function to end the conversation."""
        # interrupt any existing generation
        self.session.interrupt()

        # generate a goodbye message and hang up
        # awaiting it will ensure the message is played out before returning
        if context.userdata.first_name:
            await self.session.generate_reply(
                instructions=f"say goodbye to {context.userdata.first_name}", allow_interruptions=False
            )
        else:
            # if the user didn't provide their name, just say goodbye
            # without using their name
            await self.session.generate_reply(
                instructions="say goodbye", allow_interruptions=False
            )

        # TODO: send message to the display to show the goodbye message

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


def write_transcription_on_shutdown(ctx: JobContext, session: AgentSession[UserData]):
    """ Adds a shutdown callback to write the transcription to a file. """

    async def write_transcript():
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")

        transcripts_path = Path(os.getenv("TRANSCRIPT_PATH", "transcripts"), ctx.room.name)

        # Save the transcript to a file
        Path.mkdir(transcripts_path, parents=True, exist_ok=True)

        # This example writes to the temporary directory, but you can save to any location
        filename = transcripts_path / Path(f"transcript_{current_date}.json")
        
        with open(filename, 'w') as f:
            json.dump(session.history.to_dict(), f, indent=2)
            
        logger.info(f"transcript for {ctx.room.name} saved to {filename}")

    ctx.add_shutdown_callback(write_transcript)


async def entrypoint(ctx: JobContext):
    logger.info("Starting Experimance agent...")
    logger.debug(f"Livekit url: {os.getenv('LIVEKIT_URL', '')}")
    logger.debug(f"Livekit token: {os.getenv('LIVEKIT_TOKEN', 'None')}")
    logger.debug(f"Livekit room: {os.getenv('LIVEKIT_ROOM', ctx.room.name)}")

    session = AgentSession[UserData](
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=openai.realtime.RealtimeModel(
            model="gpt-4o-mini-realtime-preview", # gpt-4o-realtime-preview
            voice="shimmer",
            ),
        tts=None,
        # no-realtime LLM
        #llm=
        #tts=None,
        #stt=
        userdata=UserData(),
    )

    write_transcription_on_shutdown(ctx, session=session)

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=IntroAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # uncomment to enable Krisp BVC noise cancellation, use NC() to not remove extraneous voices
            noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":

    # some weird bug running console mode with uv causes the terminal 
    # to stop echoing your typing (invisible typing), so we deal with that here
    import sys, termios, atexit, os
    # grab the “before” state of console
    fd = sys.stdin.fileno()
    orig = termios.tcgetattr(fd)
    atexit.register(lambda: termios.tcsetattr(fd, termios.TCSADRAIN, orig))

    logger.setLevel(logging.DEBUG)

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))


