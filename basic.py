import modal
import subprocess
import os

app = modal.App("structure_training")

cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git", "gcc-11", "g++-11", "clang-11", "python3-dev")
    .run_commands(
        "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 "
        "--slave /usr/bin/g++ g++ /usr/bin/g++-11",
        "apt update",
        "DEBIAN_FRONTEND=noninteractive apt install -y clang-11 python3-dev",
        "pip install --upgrade pip",
        "pip install --force-reinstall 'setuptools<70' wheel",
        "pip install uv wandb",
        "git clone https://github.com/THUDM/ImageReward.git /tmp/ImageReward",
        "sed -i 's/import pkg_resources/from importlib.metadata import version as pkg_version/' /tmp/ImageReward/setup.py || true",
        "cd /tmp/ImageReward && pip install --no-build-isolation .",
    )
    .add_local_dir(
        local_path="/Users/sfkost/research/DC-Gen",
        remote_path="/DC-Gen",
        copy=True,
        ignore=[".venv", "__pycache__", "*.pyc", ".git"],
    )
    .run_commands(
        "cd /DC-Gen && uv pip install --system -e .",
    )
)

volume = modal.Volume.from_name("training_data")


def run_and_stream(cmd: list[str], env: dict[str, str]) -> None:
    print(f"Running: {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

@app.function(
    gpu="L4",
    image=cuda_image,
    secrets=[modal.Secret.from_name("wandb-key")],
    volumes={"/data": volume},
    serialized=False,
    timeout=3600 * 4,
    retries=0
)
def run_training():
    os.makedirs("/data/checkpoints", exist_ok=True)
    os.makedirs("/data/model_cache", exist_ok=True)

    env = os.environ.copy()
    env["TORCH_HOME"] = "/data/model_cache"
    env["MONAI_HOME"] = "/data/model_cache"
    env["XDG_CACHE_HOME"] = "/data/model_cache"
    env["PYTHONUNBUFFERED"] = "1"
    env["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
    env["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

    # subprocess.run(["python3", "/DC-Gen/structure.py"], check=True, env=env)
    run_and_stream([
        "python3",
        "-u",
        "/DC-Gen/config_search_3d.py",
        "--device", "cuda",
        "--dtype", "bfloat16",
        "--memory-mode", "train",
        "--no-compile",
        "--csv-path", "config_search_3d_train_bf16.csv"
    ], env=env)
    run_and_stream(["ls", "/data/checkpoints"], env=env)

@app.local_entrypoint()
def main():
    run_training.remote()
