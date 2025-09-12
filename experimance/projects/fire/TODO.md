# TODO:
- [x] pop on blur out
- [x] cancel entire queue of render requests when new story arrives
- [x] high pass filter on environmental audio
- fire crackle sound in agent [works in linux only]
- waiting for audio causing infinite renders?
  - need to always plays audio and not wait for it
- handheld mic support
- [x] cartesia timeout
- conversation ending?
- [x] "Triggering proactive greeting from backend" when no backend
- [x] send presence to core, to turn off audio
- 

# TODO LATER:
- refactor generator.py and audio_generator to share a base class
- pipecat resampling bug
- switch to user services on linux:
  - systemctl --user status 'experimance@{project}.target'
  - Move all service files to ~/.config/systemd/user/
  - Update all scripts to use systemctl --user