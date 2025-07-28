# Install Checklist


## Testing

### Projector
- [ ] keystone
- [ ] zoom
- [ ] colors
- [ ] brightness
- [ ] keeps settings after restart
- [ ] focus


### Settings

#### core:
  - [ ] `debug_depth` with display to check depth alignment
  - [ ] small touches register, but noise doesn't:
    - [ ] `change_threshold`
    - [ ] `significant_change_threshold`
  - [ ] era progression
    - [ ] `entire_surface_intensity`
    - [ ] `interaction_threshold`
    - [ ] `interaction_modifier`
    - [ ] large change == progress
    - [ ] ~8-9 small changes == progress
    - [ ] small changes go to future
    - [ ] large changes go to dystopia
    - [ ] era progresses on its own
      - [ ] `era_max_duration`
    - [ ] era stays at least X seconds
      - [ ] `era_min_duration`
  - [ ] hold hand in front of depth cam on start, remove, works


#### agent:
 - [ ] presence checks
   - [ ] detect 1 person enter/leave
   - [ ] detect multiple people enter/leave
 - [ ] interactive tuner: uv run scripts/tune_detector.py
   - [ ] brightness, etc in `profiles/face_detection.toml`
   - [ ] which detectors active (face only?)
   - [ ] cpu_audience_detector
     - [ ] `stable_readings_required`
       - [ ] Time to get stable readings (currently hardcoded at 0.8s)
     - [ ] face detector settings
       - [ ] score_threshold, etc
       - [ ] `min_face_size` to limit range of detection
 - [ ] conversation end/start
   - [ ] says hi when approach
   - [ ] search when leave
     - [ ] then goodbye
   - [ ] goodbye when goodbye
   - [ ] goodbye then leave and return, should start a new
   - [ ] goodbye, then say hello [todo]
