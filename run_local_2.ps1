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

$image1 = Join-Path $root 'project_3\outputs\part1_last.jpg'
if (-not (Test-Path $image1)) {
    throw "Input image not found at: $image1"
}

$PROMPTFILE = Join-Path $root 'project_3\prompt_2.txt'

# Run the generation script with specified parameters. You can modify the prompt, number of frames, and output filename as needed.
& $python $script `
    --strategy auto_offload `
    --prompt-file $PROMPTFILE `
    --image $image1 `
    --frames 345 `
    --width 704 `
    --height 384 `
    --steps 35 `
    --seed 22 `
    --output-dir .\project_3\outputs `
    --output "part2.mp4" `
    @args

exit $LASTEXITCODE
