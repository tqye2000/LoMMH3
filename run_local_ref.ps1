$ErrorActionPreference = 'Stop'

# Force UTF-8 for this session so non-ASCII (e.g. Chinese) prompts are not
# mangled by the console code page. VS Code's terminal already uses UTF-8, which
# is why this only failed in an external terminal.
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$script = Join-Path $root 'LoMMH.py'

if (-not (Test-Path $python)) {
    throw "Python environment not found at: $python"
}

if (-not (Test-Path $script)) {
    throw "Generation script not found at: $script"
}

# Ensure CUDA allocator uses expandable segments regardless of the shell
# environment. This prevents fragmentation OOMs that occur when a terminal
# inherits a different PYTORCH_CUDA_ALLOC_CONF value.
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

# Reference image for image-to-video generation (Ref2VA). You can replace this with your own image.
$image1 = Join-Path $root 'project_3\outputs\part1_last.jpg'
if (-not (Test-Path $image1)) {
    throw "Input image not found at: $image1"
}

$image2 = Join-Path $root 'project_3\male_role.jpg'
if (-not (Test-Path $image2)) {
    throw "Input image not found at: $image2"
}

$PROMPTFILE = Join-Path $root 'project_3\prompt_2.txt'

#$PROMPT = "The subject is posing for a photograph, then cheerfully looking at the photographer and laughing loudly, in a lively and joyful atmosphere, with a sense of movement and energy, capturing the essence of a happy moment in time."

# Write the prompt to a UTF-8 file and pass it by path. This avoids sending
# non-ASCII text across the PowerShell -> native-exe argument boundary, which is
# where the code-page corruption (and the resulting OOM) happened.
if (Test-Path variable:PROMPT) {
    $PROMPTFILE = Join-Path $root 'prompts\_prompt.txt'
    [System.IO.File]::WriteAllText($PROMPTFILE, $PROMPT, [System.Text.UTF8Encoding]::new($false))
} elseif (-not (Test-Path $PROMPTFILE)) {
    throw "Prompt variable is not defined or prompt file was not found at: $PROMPTFILE"
}

& $python $script `
    --strategy auto_offload `
    --reference-image $image1 `
    --prompt-file $PROMPTFILE `
    --frames 345 `
    --width 704 `
    --height 384 `
    --steps 35 `
    --seed 22 `
    --output-dir .\project_3\outputs `
    --output "part2.mp4" `
    @args

exit $LASTEXITCODE
