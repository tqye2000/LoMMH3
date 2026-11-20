# How to edit videos with MiniMax-H3 (`ref2va`)

This guide shows how to use a source clip as `<Video 1>` in the `ref2va` mode of [LoMMH.py](../LoMMH.py) to edit it: changing a character's look, turning a person into an animal, extending the clip, and more.

> **Status:** these recipes follow the official [full-reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md). Only Example 1 has been verified locally; expect to iterate on seed, prompt detail and reference quality for the others.

> **Known limitation: face replacement does not work.** Replacing a person in a source video with the face of someone in a reference image failed in every local test (free-form and structured prompts, `video editing` with `fully_preserved` and `attribute_transfer`). The model either reproduced the source frame for frame, or applied only the edits written in text (e.g. a new hairstyle) while keeping the original face. This was tested with the local H3-Base int8 setup, without MiniMax's hosted prompt rewriter (H3-Context-IR), and with a small reference image. The hosted MiniMax service may behave differently. To put a specific person into a scene, generate the scene from scratch with reference images instead (see [Reference media and `ref2va`](../README.md#reference-media-and-ref2va)).

In short, `video editing` reliably keeps the source's identities, framing and timing and follows *text* changes. Do not count on it taking a person's identity from a reference image.

## 1. Requirements

- The `ref2va` weights (`transformer_ref`); see [Additional `ref2va` weights](../README.md#additional-ref2va-weights).
- WSL2 with the GPU venv (`~/minimax-venv`) for `context_parallel`, or `--strategy auto_offload` on a single GPU.
- A source clip of 2–15 s. The model works at 24 fps; other rates are resampled.
- Reference images (for scenes, props or styles): large, sharp, and tightly cropped on what they should provide.

## 2. Running an edit

The launcher [run_ref2va_video.sh](../run_ref2va_video.sh) takes one reference image and one reference video. Override its inputs with environment variables:

| Variable | Meaning |
| --- | --- |
| `REF_VIDEO` | Source clip (`<Video 1>`) |
| `REF_IMAGE` | Reference image (`<Picture 1>`) |
| `PROMPTFILE` | UTF-8 prompt file |
| `OUTPUT_DIR`, `OUTPUT` | Output folder and MP4 name |

Extra arguments are passed on to `LoMMH.py`:

```powershell
wsl -d Ubuntu -- bash -lc 'REF_VIDEO=/mnt/d/TQYE/Experiments/MiniMax/project_3/outputs/part1.mp4 REF_IMAGE=/mnt/d/TQYE/Experiments/MiniMax/images/cl.jpg PROMPTFILE=/mnt/d/TQYE/Experiments/MiniMax/project_4/prompt_video_v4_edit.txt OUTPUT=part1_edit.mp4 bash /mnt/d/TQYE/Experiments/MiniMax/run_ref2va_video.sh --seed 7'
```

For other combinations (several images, or no image at all), call `LoMMH.py` directly from WSL:

```bash
cd /mnt/d/TQYE/Experiments/MiniMax
~/minimax-venv/bin/python LoMMH.py --strategy context_parallel \
  --reference-image images/my_cat.jpg \
  --reference-video project_3/outputs/part1.mp4 \
  --prompt-file project_4/prompt_animal.txt \
  --frames 345 --width 704 --height 384 --steps 35 --seed 22 \
  --reference-video-short-edge 384 \
  --output-dir project_4/outputs --output part1_animal.mp4
```

Key options:

- `--frames`: for an edit, use the source clip's length. The reference video is cut to the generated frame count.
- `--width` / `--height`: match the source clip's aspect ratio.
- `--reference-video-short-edge`: reference videos are resized to a 768 px short edge by default, regardless of the output size. Lower it (e.g. `384` for a 704×384 clip) if the conditioner runs out of memory.
- `--reference-short-edge`: short edge for reference images (default 2048).

### Reference labels

The model numbers references by type, in command-line order:

- `--reference-image` → `<Picture 1>`, `<Picture 2>`, …
- `--reference-video` → `<Video 1>`, …; if the clip has sound, its soundtrack is also passed as `<Audio 1>`, …

## 3. Prompt format

Write the prompt in English, with these six sections in order. Keep dialogue in its original language inside `<d>[Language] ...</d>`.

| Section | Purpose |
| --- | --- |
| `subject_definitions:` | One line per label: what `<Subject N>`, `<Video N>` and `<Audio N>` are and where each comes from |
| `summary:` | Task type in brackets, then one short paragraph about the target video |
| `retention_analysis:` | How each label is kept: `fully_preserved`, `partially_preserved`, `attribute_transfer`, `weak_reference`; for audio `fully_copy`, `partially_copy`, `reference`, `weak_reference` |
| `detailed_description:` | A style sentence, then `[Shot 1] ...`, `[Shot 2] At 00:04.500, ...` in playback order |
| `overall_soundscape:` | Ambience and physical sounds |
| `non_diegetic_music:` | Background score, or `N/A` |

Task types (combine them with ` + `):

| Task type | Use when |
| --- | --- |
| `video editing` | The source video itself is modified |
| `video continuation` | New content continues from the end of the source video |
| `reference generation` | A reference gives a character, style, motion or camera guidance only |
| `keyframe completion` | An image is a concrete first, last or key frame |
| `audio reuse` | The source audio is copied in whole or in part |
| `audio reference` | Only voice timbre, style or rhythm is referenced |

Rules of thumb:

- `<Subject N>` is something visible that is reused or changed: a person, an animal, a scene or a prop. Describe its looks the first time it appears in `detailed_description`.
- `<Video 1>` is the edit source: `<Video 1> is the source video for the target video edit.`
- For editing, start the summary with `The target video is an edited version of <Video 1>.`
- Give each speaker a stable ID: `<Subject 1> (S1) says, <d>[Chinese] ...</d>`.
- Match the shot cut times and dialogue to the real source clip. Use [save_video_frames.py](../save_video_frames.py) or a player to check them.

## 4. Examples

The examples use a 14.4 s source clip ([project_3/outputs/part1.mp4](../project_3/outputs/part1.mp4)). A woman (left) and a man (right) share red wine in a hotel lounge at night. Its hard cuts are at 5.500 s, 9.000 s and 12.375 s:

1. Over the man's right shoulder onto the woman, left of center.
2. The reverse angle, over her left shoulder onto the man.
3. Back to the framing of shot 1.
4. A wide two-shot across the round table.

### Example 1: Change a character's look with text

Inputs: `--reference-video source.mp4`. A reference image is optional, but its face is not transferred (see the limitation above).

Describe the change in words and keep everything else. This was verified locally: the woman's long wavy hair became short chin-length hair, while her face, the man, the framing, the cuts and the soundtrack stayed as in the source. The full file is [project_4/prompt_video_v4_edit.txt](../project_4/prompt_video_v4_edit.txt). Abridged, with the face-transfer parts removed:

```text
subject_definitions:
<Subject 1> is the woman in <Video 1>, wearing a wine-red velvet spaghetti-strap evening dress, red lipstick, and drop earrings; in the target video her long wavy hair is changed to short dark chin-length hair with side-swept bangs.
<Subject 2> is the man in <Video 1>, in his early thirties, with short neat dark hair, a dark suit jacket, and a black shirt.
<Video 1> is the source video for the editing task.
<Audio 1> is the synchronized audio track of <Video 1>, providing both voices and the room ambience.

summary:
[video editing + audio reuse] The target video is an edited version of <Video 1>. The woman's long wavy hair is changed to short dark chin-length hair with side-swept bangs; her face, dress, pose, actions, and lines stay as in <Video 1>. <Subject 2>, the hotel lounge, the framing, the camera movement, and the cuts at 00:05.500, 00:09.000, and 00:12.375 are unchanged. <Audio 1> is reused as the complete soundtrack.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3], [Shot 4]): partially_preserved - her face, dress, jewelry, pose, and movements are retained; her hair is changed to a short dark chin-length cut.
<Subject 2> (appears in [Shot 1], [Shot 2], [Shot 3], [Shot 4]): fully_preserved - his face, hair, suit, position, and performance are retained.
<Video 1> (source video editing): partially_preserved - the lounge, lighting, framing, camera movement, cut timing, props, and all actions are retained, while the woman's hair is changed.
<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.

detailed_description:
The target video is a realistic, cinematic drama with film grain and low-key lighting: warm amber lamps and candles inside, a cold blue city skyline through the window.
[Shot 1] The shot follows <Video 1>: a medium shot over the right shoulder of <Subject 2> ... <Subject 1> (S1) sits left of center on a dark red velvet sofa ... with short dark chin-length hair with side-swept bangs, not long wavy hair ... says, <d>[Chinese] 夜这么深，先生还肯来，倒叫我意外。</d>
[Shot 2] At 00:05.500, the shot cuts to the reverse angle over the left shoulder of <Subject 1>: her short dark hair and bare shoulder are blurred in the left foreground ...
[Shot 3] At 00:09.000, the shot cuts back to the framing of [Shot 1] ...
[Shot 4] At 00:12.375, the shot cuts to a wide two-shot across the round table ...

overall_soundscape:
The quiet room tone, low city hum, and soft clink of glass from <Audio 1> continue throughout the target video.

non_diegetic_music:
N/A
```

Tips:

- Name the old and the new state ("not long wavy hair") in every shot the person appears in.
- Clearly visible changes such as hair, clothing, makeup or accessories work. A different face does not.
- For a new voice, change `<Audio 1>` to `partially_copy` (keep the ambience, replace the lines). Or add a voice sample: `<Audio 2> is the voice-timbre reference for <Subject 1> (S1).` This needs a reference video whose soundtrack carries that voice; LoMMH.py has no audio-only option yet.

### Example 2: Turn a person into a personalised animal

Inputs: `--reference-image my_cat.jpg --reference-video source.mp4`

Here the man becomes the user's own cat, a ginger tabby, in the same suit and pose. Use `attribute_transfer` for what moves from him to the cat.

Not verified. Given the face-replacement results, the cat's exact markings may come more from the text than from the photo, so describe the fur, eyes and markings in detail in every shot.

```text
subject_definitions:
<Subject 1> is the ginger tabby cat in <Picture 1>, with orange-and-cream striped fur, a white chest and muzzle, amber eyes, and a pink nose. It provides the identity and fur pattern of the male lead.
<Subject 2> is the woman in <Video 1>, with long wavy dark hair, a wine-red velvet evening dress, red lipstick, and delicate earrings.
<Video 1> is the source video for the target video edit.
<Audio 1> is the synchronized audio track of <Video 1> and is reused in the target video.

summary:
[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>. The man seated at frame-right in <Video 1> is replaced by <Subject 1>, an anthropomorphic ginger tabby cat who sits upright in his chair, wears his dark fitted suit and black shirt, holds his wine glass in a paw, and performs his gestures and lines. <Subject 2>, the lounge, the camera work, the cuts, and the lighting stay as in <Video 1>, and <Audio 1> is reused as the complete soundtrack.

retention_analysis:
<Subject 1> (appears in [Shot 2], [Shot 4]): fully_preserved - the cat's face, striped orange-and-cream fur, white chest, amber eyes, and pink nose from <Picture 1> are retained.
<Video 1> (the man's suit, pose, gestures, and timing): attribute_transfer - the man's dark suit, black shirt, seated posture, glass-turning gesture, eyeline, and lip movement are transferred to <Subject 1>.
<Subject 2> (appears in [Shot 1], [Shot 3], [Shot 4]): fully_preserved - her face, hair, dress, position, and performance are retained.
<Video 1> (source video): partially_preserved - everything is retained except the man at frame-right.
<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.

detailed_description:
The target video is a realistic, cinematic drama with film grain and low-key lighting; the cat is rendered photorealistically, with natural fur detail lit by the scene.
[Shot 1] As in <Video 1>, a medium shot over the right shoulder of <Subject 1>, whose orange-striped ears and the back of its suit are blurred at the right edge, onto <Subject 2> (S1) left of center ... says, <d>[Chinese] 夜这么深，先生还肯来，倒叫我意外。</d>
[Shot 2] At 00:05.500, the shot cuts to the reverse angle over her left shoulder onto <Subject 1> (S2), the ginger tabby cat from <Picture 1>, sitting upright on the far side of the table in a dark suit and black shirt. The cat turns the wine glass half a turn with one paw, raises its amber eyes to her, and its mouth moves in sync as it says in a low, slow male voice, <d>[Chinese] 你来，我为什么不来。只是想知道，这回你约我，是想送我上路，还是想留我。</d> Its whiskers twitch slightly; its expression stays calm.
[Shot 3] At 00:09.000, ...
[Shot 4] At 00:12.375, a wide two-shot across the round table: <Subject 2> on the left and <Subject 1>, the suited ginger cat, on the right, looking at each other in silence ...

overall_soundscape:
The room tone and clink of glass copied from <Audio 1> continue throughout.

non_diegetic_music:
N/A
```

Tips:

- Decide clearly between a real cat (on four legs, no speech) and an anthropomorphic one (clothes, paws, speech), and describe it consistently in every shot.
- For a non-speaking animal, remove its lines and use `partially_copy` for `<Audio 1>`. Describe the new sounds, e.g. a soft purr.
- Use several photos of the pet (`<Picture 1>`, `<Picture 2>`, …) and write `<Subject 1> is the cat in <Picture 1> and <Picture 2>, ...`.

### Example 3: Extend the original video

Inputs: `--reference-video source.mp4`, and optionally `--reference-image` for a new character.

The output is a new clip of up to 15 s that starts where the source ends. Join the two afterwards (see below).

```text
subject_definitions:
<Subject 1> is the woman in <Video 1>, with long wavy dark hair, a wine-red velvet evening dress, red lipstick, and delicate earrings.
<Subject 2> is the man in <Video 1>, in his early thirties, with short neat dark hair, a dark fitted suit, and a black shirt.
<Subject 3> is the private hotel lounge in <Video 1>, with velvet seats, a round marble table, warm wall lamps, a low crystal chandelier, and a cold blue city skyline through the window.
<Video 1> is the source video that the target video continues from its final frame.
<Audio 1> is the synchronized audio track of <Video 1>, providing the voices of <Subject 1> (S1) and <Subject 2> (S2) and the room ambience.

summary:
[video continuation + audio reference] The target video continues directly from the final frame of <Video 1>: the two-shot of <Subject 1> and <Subject 2> facing each other across the table in <Subject 3>. <Subject 2> breaks the silence, <Subject 1> answers and rises from her seat, and the scene ends on her walking to the window. The voices and ambience continue those of <Audio 1> without copying the signal.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - her face, hair, dress, and makeup are retained.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - his face, hair, and suit are retained.
<Subject 3> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the lounge, its furniture, and the warm amber and cold blue lighting are retained.
<Video 1> (continuation source): fully_preserved - the target video begins in exactly the state, framing, and lighting of the final frame of <Video 1>.
<Audio 1>: reference - the voice timbre of <Subject 1> (S1) and <Subject 2> (S2) and the quiet room tone follow <Audio 1> without copying the original signal.

detailed_description:
The target video is a realistic, cinematic drama with film grain, low-key lighting, and strong contrast between warm amber interior light and cold blue city light.
[Shot 1] The shot continues seamlessly from the final frame of <Video 1>: a 50mm two-shot in <Subject 3>, with <Subject 1> at frame-left and <Subject 2> at frame-right ... <Subject 2> (S2) leans forward and says in a low, slow voice, <d>[Chinese] 那你告诉我，今晚这杯酒，是敬谁的？</d>
[Shot 2] At 00:05.000, the shot cuts to an 85mm medium close-up of <Subject 1> (S1) at frame-left ... she says softly, <d>[Chinese] 敬一个不肯说真话的人。</d> She rises from her seat.
[Shot 3] At 00:10.000, a wide shot: <Subject 1> walks slowly to the window and stops with her back to the camera, silhouetted against the blue city lights, while <Subject 2> stays seated, watching her. The camera holds still.

overall_soundscape:
Quiet room tone, a low city hum, the rustle of fabric as she stands, and soft footsteps on carpet.

non_diegetic_music:
N/A
```

Run it with the image check removed, by calling `LoMMH.py` directly:

```bash
~/minimax-venv/bin/python LoMMH.py --strategy context_parallel \
  --reference-video project_3/outputs/part1.mp4 \
  --prompt-file project_4/prompt_extend.txt \
  --frames 345 --width 704 --height 384 --steps 35 --seed 22 \
  --reference-video-short-edge 384 \
  --output-dir project_4/outputs --output part1_ext.mp4
```

Then join the source and the extension:

```powershell
& '.venv\Scripts\python.exe' join_videos_with_audio.py `
  project_3\outputs\part1.mp4 project_4\outputs\part1_ext.mp4 project_4\outputs\part1_extended.mp4
```

Tips:

- The source and the extension together are limited only by what you join; each generated clip is 5–15 s.
- If the first frame of the extension does not line up with the source's last frame, try the first-frame mode instead. Save the last frame with `save_video_frames.py`, then run `fl2va` with `--image <last frame>` (no `--reference-video`). This pins the first frame exactly, but loses the motion and voice reference.
- Adding `--reference-image` for a new character works the same way: define `<Subject 4> is the man in <Picture 1>, ...` and use `[video continuation + reference generation + audio reference]`.

### Example 4: Change the scene or style, keep the action

Inputs: `--reference-image beach.jpg --reference-video source.mp4`

Not verified. Describe the new scene fully in text as well as in the picture.

```text
subject_definitions:
<Subject 1> is the woman in <Video 1>, with long wavy dark hair, a wine-red velvet evening dress, and red lipstick.
<Subject 2> is the man in <Video 1>, with short neat dark hair, a dark fitted suit, and a black shirt.
<Subject 3> is the sunset beach terrace in <Picture 1>, with a wooden deck, white linen chairs, string lights, and a golden sea horizon.
<Video 1> is the source video for the target video edit.
<Audio 1> is the synchronized audio track of <Video 1>; its dialogue is reused.

summary:
[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>. The hotel lounge is replaced by <Subject 3>, lit by warm golden-hour sunlight; <Subject 1>, <Subject 2>, their positions, actions, lines, the camera work, and the cuts stay as in <Video 1>. The dialogue of <Audio 1> is reused and its room tone is replaced by gentle waves.

retention_analysis:
<Subject 1>, <Subject 2>: fully_preserved - faces, hair, clothing, positions, and performances are retained.
<Subject 3> (appears in all shots): fully_preserved - the deck, chairs, string lights, and sea horizon are retained.
<Video 1> (source video): partially_preserved - characters, actions, framing, camera movement, and cut timing are retained; the background and lighting are replaced.
<Audio 1>: partially_copy - the dialogue is copied; the indoor ambience is replaced by sea sounds.
...
```

The same pattern covers other edits:

- **Restyling:** e.g. `The target video is rendered in a hand-drawn 2D anime style` with `<Picture 1>` as a style reference: `<Subject 3> is the visual style of <Picture 1>, ...` and `attribute_transfer`.
- **Wardrobe or prop swap:** `<Subject 3> is the white tuxedo in <Picture 1>, ...`, with `partially_preserved` for the person who wears it.

## 5. Troubleshooting

| Problem | What to try |
| --- | --- |
| Output copies the source frame for frame | Use the structured format above, with `partially_preserved` or `attribute_transfer` for whatever changes. Describe each change in text in every shot. Check that the shots and cut times match the real clip. |
| A person's face is not replaced by the reference image | Known limitation (see the top of this guide). Generate the scene from scratch with reference images instead. |
| CUDA OOM during "Encoding prompt" | Lower `--reference-video-short-edge` (e.g. 384), lower `--reference-short-edge`, reduce `--frames`, or use `--strategy auto_offload`. |
| Cuts or lip sync drift from the source | Check the shot times in `detailed_description` against the real clip; keep `--frames` equal to the source length. |
| Wrong voice for the new character | Do not use `fully_copy` for the audio; reference a voice sample instead. |
| Extension does not match the last frame | Use the `fl2va` first-frame route described in Example 3. |

See also the [README](../README.md#video-editing-with-ref2va) and the [MiniMax-H3 full-reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
