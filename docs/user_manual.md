# Smart Glasses User Manual

This manual explains how to use the current running application in `main.py`.

## 1. How Interaction Works

1. System starts and opens camera view.
2. User says wake word: `vision`.
3. System responds and listens for one command.
4. System performs the command and speaks the result.
5. System returns to wake-word listening state.

## 2. Core Commands

### Detect objects

- Say: `detect`
- Result: object detection summary is spoken.

### Read text on objects

- Say: `read`
- Result: objects are detected, OCR is run on detected crops, and text-aware summary is spoken.

### Repeat last response

- Say: `repeat`
- Result: last spoken output is repeated.

## 3. Directional Commands

You can target scene regions using direction words:

- `left`
- `front`
- `right`

Examples:

- `detect left`
- `read front`
- `detect right`

## 4. Session and Exit Commands

- `sleep`
- `end`
- `nevermind`
- `thanks`

These commands return the system to idle/wake-word behavior for the next interaction cycle.

## 5. Help Commands

- `commands`
- `command`
- `help`

After help output, the system may ask whether to play directional/exit command guidance.

- Say `yes` to hear extended guidance.
- Say `no` to skip it.

## 6. Startup Tutorial

On startup, the app may read `tutorial.txt`.

Related command help text files:

- `commands_user_facing.txt`
- `commands_directional_exit.txt`

## 7. Visual Windows

The application typically shows two OpenCV windows:

- Live camera feed
- Detections/annotations

Press `q` in the OpenCV window to quit.

## 8. Practical Usage Tips

- Pause briefly after saying `vision`, then say your command clearly.
- Use short command phrases for best recognition accuracy.
- Keep objects within camera view and reasonably well lit.
- For OCR (`read`), hold text steady and within clear focus.

## 9. Known Current Limitations

- Command grammar is constrained to supported terms.
- OCR quality depends on lighting, angle, and text size.
- Some dependencies (`vosk`, `sounddevice`, `inflect`) may require manual install depending on environment setup.
