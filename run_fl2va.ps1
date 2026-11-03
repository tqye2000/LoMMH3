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

$first_image = Join-Path $root 'images\paradise.jpg'
if (-not (Test-Path $first_image)) {
    throw "Input image not found at: $first_image"
}

#$PROMPTFILE = Join-Path $root 'prompt.txt'
$PROMPTFILE = Join-Path $root 'prompts\paradise.txt'

# Run the generation script with specified parameters. You can modify the prompt, number of frames, and output filename as needed.
& $python $script `
    --strategy auto_offload `
    --prompt-file $PROMPTFILE `
    --image $first_image `
    --frames 345 `
    --width 1024 `
    --height 1024 `
    --steps 35 `
    --seed 22 `
    --output-dir .\outputs `
    --output "paradise.mp4" `
    @args

exit $LASTEXITCODE
