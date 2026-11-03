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

# Number of GPUs to split one clip across (context parallel). All four RTX 6000
# Ada cards participate by default; override by passing -Gpus <n> before @args or
# editing this value.
$GPUS = 4

# Ensure CUDA allocator uses expandable segments regardless of the shell
# environment. This prevents fragmentation OOMs that occur when a terminal
# inherits a different PYTORCH_CUDA_ALLOC_CONF value.
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

# The Windows PyTorch wheels are built without libuv. LoMMH.py builds its own
# rendezvous store with use_libuv=False, but keep this set as a belt-and-braces
# default for any other TCPStore the stack creates.
$env:USE_LIBUV = '0'

# Number of GPUs to split one clip across. LoMMH.py spawns one worker per GPU.
$env:CP_WORLD_SIZE = $GPUS

# Reference images for image-to-video generation (Ref2VA). Replace with your own.
$image1 = Join-Path $root 'project_2\hl_3.jpg'
if (-not (Test-Path $image1)) {
    throw "Input image not found at: $image1"
}

$image2 = Join-Path $root 'project_2\chongda_a.jpg'
if (-not (Test-Path $image2)) {
    throw "Input image not found at: $image2"
}

$PROMPTFILE = Join-Path $root 'project_2\prompt.txt'

# Write the prompt to a UTF-8 file and pass it by path. This avoids sending
# non-ASCII text across the PowerShell -> native-exe argument boundary, which is
# where the code-page corruption (and the resulting OOM) happened.
if (Test-Path variable:PROMPT) {
    $PROMPTFILE = Join-Path $root 'prompts\_prompt.txt'
    [System.IO.File]::WriteAllText($PROMPTFILE, $PROMPT, [System.Text.UTF8Encoding]::new($false))
} elseif (-not (Test-Path $PROMPTFILE)) {
    throw "Prompt variable is not defined or prompt file was not found at: $PROMPTFILE"
}

# Context parallelism splits a single clip's denoise across every GPU. LoMMH.py
# spawns one worker process per GPU itself (via torch.multiprocessing.spawn) and
# builds an explicit non-libuv rendezvous store, so it is launched as a plain
# python process rather than through torchrun (whose agent store cannot skip
# libuv on the Windows wheels).
& $python `
    $script `
    --strategy context_parallel `
    --reference-image $image1 `
    --reference-image $image2 `
    --prompt-file $PROMPTFILE `
    --frames 345 `
    --width 704 `
    --height 384 `
    --steps 30 `
    --seed 42 `
    --output-dir .\project_2\outputs `
    --output "hl_part1_cp.mp4" `
    @args

exit $LASTEXITCODE
