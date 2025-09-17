# TODO:
- [x] pop on blur out
- [x] cancel entire queue of render requests when new story arrives
- [x] high pass filter on environmental audio
- fire crackle sound in agent [works in linux only]
- [?] waiting for audio causing infinite renders?
  - need to always plays audio and not wait for it
- [x]handheld mic support
- [x] cartesia timeout
- conversation ending?
- [x] "Triggering proactive greeting from backend" when no backend
- [x] send presence to core, to turn off audio
- [x] prompt saving
- [x] transcript+prompt streaming
  - [x] order/display by time
- [ ] image streaming for base images only
  - [ ] combine with transcript streaming
- [x] smart plug: 1596-041-7619


# TODO LATER:
- refactor generator.py and audio_generator to share a base class
- pipecat resampling bug
- switch to user services on linux:
  - systemctl --user status 'experimance@{project}.target'
  - Move all service files to ~/.config/systemd/user/
  - Update all scripts to use systemctl --user