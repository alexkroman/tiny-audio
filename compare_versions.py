#!/usr/bin/env python3
"""Compare Poetry lock file versions with server installed packages."""

import re
from pathlib import Path

# Server packages from user input
SERVER_PACKAGES = """accelerate                1.10.1
aiofiles                  24.1.0
aiohappyeyeballs          2.6.1
aiohttp                   3.13.0
aiosignal                 1.4.0
annotated-types           0.7.0
antlr4-python3-runtime    4.9.3
anyio                     4.11.0
argon2-cffi               25.1.0
argon2-cffi-bindings      25.1.0
arrow                     1.3.0
assemblyai                0.44.3
asttokens                 3.0.0
async-lru                 2.0.5
attrs                     25.4.0
babel                     2.17.0
beautifulsoup4            4.14.2
bleach                    6.2.0
blinker                   1.7.0
Brotli                    1.1.0
build                     1.3.0
CacheControl              0.14.3
certifi                   2025.10.5
cffi                      2.0.0
charset-normalizer        3.4.4
cleo                      2.1.0
click                     8.3.0
comm                      0.2.3
crashtest                 0.4.1
cryptography              41.0.7
datasets                  4.2.0
dbus-python               1.3.2
debugpy                   1.8.17
decorator                 5.2.1
defusedxml                0.7.1
dill                      0.4.0
distlib                   0.4.0
distro                    1.9.0
dulwich                   0.24.5
einops                    0.8.1
evaluate                  0.4.6
executing                 2.2.1
fastapi                   0.119.0
fastjsonschema            2.21.2
ffmpy                     0.6.3
filelock                  3.20.0
findpython                0.7.0
flash_attn                2.8.3
fqdn                      1.5.1
frozenlist                1.8.0
fsspec                    2025.9.0
gitdb                     4.0.12
GitPython                 3.1.45
gradio                    5.49.1
gradio_client             1.13.3
groovy                    0.1.2
h11                       0.16.0
hf_transfer               0.1.9
hf-xet                    1.1.10
httpcore                  1.0.9
httplib2                  0.20.4
httpx                     0.28.1
huggingface-hub           0.35.3
hydra-core                1.3.2
idna                      3.11
installer                 0.7.0
ipykernel                 6.30.1
ipython                   9.6.0
ipython_pygments_lexers   1.1.1
ipywidgets                8.1.7
isoduration               20.11.0
jaraco.classes            3.4.0
jaraco.context            6.0.1
jaraco.functools          4.3.0
jedi                      0.19.2
jeepney                   0.9.0
Jinja2                    3.1.6
jiwer                     4.0.0
json5                     0.12.1
jsonpointer               3.0.0
jsonschema                4.25.1
jsonschema-specifications 2025.9.1
jupyter-archive           3.4.0
jupyter_client            8.6.3
jupyter_core              5.8.1
jupyter-events            0.12.0
jupyter-lsp               2.3.0
jupyter_server            2.17.0
jupyter_server_terminals  0.5.3
jupyterlab                4.4.9
jupyterlab_pygments       0.3.0
jupyterlab_server         2.27.3
jupyterlab_widgets        3.0.15
keyring                   25.6.0
lark                      1.3.0
launchpadlib              1.11.0
lazr.restfulclient        0.14.6
lazr.uri                  1.0.6
markdown-it-py            3.0.0
MarkupSafe                3.0.3
matplotlib-inline         0.1.7
mdurl                     0.1.2
mistune                   3.1.4
more-itertools            10.8.0
mpmath                    1.3.0
msgpack                   1.1.2
multidict                 6.7.0
multiprocess              0.70.16
nbclient                  0.10.2
nbconvert                 7.16.6
nbformat                  5.10.4
nest-asyncio              1.6.0
networkx                  3.5
ninja                     1.13.0
notebook                  7.4.2
notebook_shim             0.2.4
numpy                     2.3.4
nvidia-cublas-cu12        12.8.4.1
nvidia-cuda-cupti-cu12    12.8.90
nvidia-cuda-nvrtc-cu12    12.8.93
nvidia-cuda-runtime-cu12  12.8.90
nvidia-cudnn-cu12         9.10.2.21
nvidia-cufft-cu12         11.3.3.83
nvidia-cufile-cu12        1.13.1.3
nvidia-curand-cu12        10.3.9.90
nvidia-cusolver-cu12      11.7.3.90
nvidia-cusparse-cu12      12.5.8.93
nvidia-cusparselt-cu12    0.7.1
nvidia-nccl-cu12          2.27.3
nvidia-nvjitlink-cu12     12.8.93
nvidia-nvtx-cu12          12.8.90
oauthlib                  3.2.2
omegaconf                 2.3.0
orjson                    3.11.3
packaging                 25.0
pandas                    2.3.3
pandocfilters             1.5.1
parso                     0.8.5
pbs-installer             2025.10.14
peft                      0.17.1
pexpect                   4.9.0
pillow                    11.3.0
pip                       25.2
pkginfo                   1.12.1.2
platformdirs              4.5.0
poetry                    2.2.1
poetry-core               2.2.1
prometheus_client         0.23.1
prompt_toolkit            3.0.52
propcache                 0.4.1
protobuf                  6.32.1
psutil                    7.1.0
ptyprocess                0.7.0
pure_eval                 0.2.3
pyarrow                   21.0.0
pycparser                 2.23
pydantic                  2.11.10
pydantic_core             2.33.2
pydub                     0.25.1
Pygments                  2.19.2
PyGObject                 3.48.2
PyJWT                     2.7.0
pyparsing                 3.1.1
pyproject_hooks           1.2.0
python-apt                2.7.7+ubuntu5
python-dateutil           2.9.0.post0
python-json-logger        4.0.0
python-multipart          0.0.20
pytz                      2025.2
PyYAML                    6.0.3
pyzmq                     27.1.0
RapidFuzz                 3.14.1
referencing               0.36.2
regex                     2025.9.18
requests                  2.32.5
requests-toolbelt         1.0.0
rfc3339-validator         0.1.4
rfc3986-validator         0.1.1
rfc3987-syntax            1.1.0
rich                      14.2.0
rpds-py                   0.27.1
ruff                      0.14.0
safehttpx                 0.1.6
safetensors               0.6.2
SecretStorage             3.4.0
semantic-version          2.10.0
Send2Trash                1.8.3
sentry-sdk                2.42.0
setuptools                80.9.0
shellingham               1.5.4
six                       1.17.0
smmap                     5.0.2
sniffio                   1.3.1
soupsieve                 2.8
stack-data                0.6.3
starlette                 0.48.0
sympy                     1.14.0
terminado                 0.18.1
tiny-audio                0.1.0          /workspace
tinycss2                  1.4.0
tokenizers                0.22.1
tomlkit                   0.13.3
torch                     2.8.0
torchaudio                2.8.0+cu128
torchcodec                0.8.0
torchvision               0.23.0+cu128
tornado                   6.5.2
tqdm                      4.67.1
traitlets                 5.14.3
transformers              4.57.1
triton                    3.4.0
trove-classifiers         2025.9.11.17
typer                     0.19.2
types-python-dateutil     2.9.0.20251008
typing_extensions         4.15.0
typing-inspection         0.4.2
tzdata                    2025.2
uri-template              1.3.0
urllib3                   2.5.0
uvicorn                   0.37.0
virtualenv                20.34.0
wadllib                   1.3.6
wandb                     0.22.2
wcwidth                   0.2.14
webcolors                 24.11.1
webencodings              0.5.1
websocket-client          1.9.0
websockets                15.0.1
widgetsnbextension        4.0.14
xxhash                    3.6.0
yarl                      1.22.0
zstandard                 0.25.0"""


def parse_server_packages():
    """Parse server packages into a dict."""
    packages = {}
    for line in SERVER_PACKAGES.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0].lower().replace('_', '-')
            version = parts[1]
            packages[name] = version
    return packages


def parse_poetry_lock():
    """Parse poetry.lock file to extract package versions."""
    lock_file = Path('poetry.lock')
    if not lock_file.exists():
        print("poetry.lock file not found!")
        return {}

    packages = {}
    current_package = None

    with open(lock_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Match package name: [[package]]
            if line.startswith('[[package]]'):
                current_package = None

            # Match name = "package-name"
            name_match = re.match(r'^name\s*=\s*"([^"]+)"', line)
            if name_match:
                current_package = name_match.group(1).lower()

            # Match version = "1.2.3"
            version_match = re.match(r'^version\s*=\s*"([^"]+)"', line)
            if version_match and current_package:
                version = version_match.group(1)
                packages[current_package] = version
                current_package = None

    return packages


def compare_versions():
    """Compare versions between server and poetry.lock."""
    server_pkgs = parse_server_packages()
    lock_pkgs = parse_poetry_lock()

    print("=" * 80)
    print("VERSION COMPARISON: Poetry Lock vs Server")
    print("=" * 80)

    # Find differences
    differences = []
    missing_in_lock = []
    missing_on_server = []

    # Check packages in lock file
    for pkg, lock_version in sorted(lock_pkgs.items()):
        if pkg in server_pkgs:
            server_version = server_pkgs[pkg]
            if lock_version != server_version:
                differences.append((pkg, lock_version, server_version))
        else:
            missing_on_server.append((pkg, lock_version))

    # Check packages on server not in lock
    for pkg, server_version in sorted(server_pkgs.items()):
        if pkg not in lock_pkgs:
            missing_in_lock.append((pkg, server_version))

    # Print results
    if differences:
        print("\n🔄 VERSION DIFFERENCES:")
        print(f"{'Package':<30} {'Poetry Lock':<20} {'Server':<20}")
        print("-" * 80)
        for pkg, lock_v, server_v in differences:
            print(f"{pkg:<30} {lock_v:<20} {server_v:<20}")

    if missing_on_server:
        print(f"\n📦 IN POETRY.LOCK BUT NOT ON SERVER ({len(missing_on_server)}):")
        print(f"{'Package':<30} {'Version':<20}")
        print("-" * 80)
        for pkg, version in missing_on_server[:20]:  # Show first 20
            print(f"{pkg:<30} {version:<20}")
        if len(missing_on_server) > 20:
            print(f"... and {len(missing_on_server) - 20} more")

    if missing_in_lock:
        print(f"\n📦 ON SERVER BUT NOT IN POETRY.LOCK ({len(missing_in_lock)}):")
        print(f"{'Package':<30} {'Version':<20}")
        print("-" * 80)
        for pkg, version in missing_in_lock[:20]:  # Show first 20
            print(f"{pkg:<30} {version:<20}")
        if len(missing_in_lock) > 20:
            print(f"... and {len(missing_in_lock) - 20} more")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  - Version mismatches: {len(differences)}")
    print(f"  - Missing on server: {len(missing_on_server)}")
    print(f"  - Missing in lock: {len(missing_in_lock)}")
    print(f"  - Total packages in lock: {len(lock_pkgs)}")
    print(f"  - Total packages on server: {len(server_pkgs)}")
    print("=" * 80)


if __name__ == "__main__":
    compare_versions()
