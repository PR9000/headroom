## Installing on FreeBSD

Headroom can be built and run natively on **FreeBSD 15.1-STABLE amd64** and later.

> [!NOTE]
> Native FreeBSD support for the `ort` crate's `load-dynamic` feature has already
> been merged into the upstream project through PR #601, but is not yet available
> in the latest published crates.io release (`v2.0.0-rc.12`). The steps below
> describe the temporary workaround required until a new `ort` release is
> published.

### Prerequisites

Ensure the following components are installed:

* **FreeBSD 15.1-STABLE amd64** or later
* **Rust** and **Cargo** (latest stable toolchain)
* **Python 3.12+**

#### Precompiled Dependencies (FreeBSD 15.1-STABLE amd64 package snapshot)

The following package list represents the precompiled dependencies available
from the official FreeBSD package repository used during native validation.

Validation environment:

```console
$ freebsd-version
15.1-STABLE

$ pkg -vv | grep url
url: pkg+https://pkg.FreeBSD.org/FreeBSD:15:amd64/latest
url: pkg+https://pkg.FreeBSD.org/FreeBSD:15:amd64/kmods_latest
url: pkg+https://pkg.FreeBSD.org/FreeBSD:15:amd64/base_latest
```

Installing these packages avoids compiling heavy Rust-backed Python dependencies
(such as tokenizers, tiktoken, and pydantic-core) from source.

```bash
sudo pkg install -y \
    cargo-make \
    make \
    rust \
    py312-aiohappyeyeballs-2.6.2 \
    py312-aiohttp-3.14.1 \
    py312-aiosignal-1.4.0 \
    py312-annotated-doc-0.0.4 \
    py312-annotated-types-0.7.0 \
    py312-anyio-4.14.0 \
    py312-attrs-26.1.0 \
    py312-certifi-2026.6.17 \
    py312-charset-normalizer-3.4.7 \
    py312-click-8.3.1 \
    py312-distro-1.9.0 \
    py312-fastuuid-0.14.0_6 \
    py312-filelock-3.29.6 \
    py312-frozenlist-1.8.0 \
    py312-fsspec-2026.6.0 \
    py312-h11-0.16.0 \
    py312-hf-xet-1.5.1 \
    py312-httpcore-1.0.9 \
    py312-httpx-0.28.1_2 \
    py312-huggingface-hub-1.16.1 \
    py312-idna-3.18 \
    py312-importlib-metadata-8.7.1 \
    py312-jinja2-3.1.6 \
    py312-jiter-0.16.0 \
    py312-jsonschema-4.26.0 \
    py312-jsonschema-specifications-2025.9.1 \
    py312-litellm-1.91.0 \
    py312-markdown-it-py-4.2.0 \
    py312-markupsafe-3.0.3 \
    py312-mdurl-0.1.2_2 \
    py312-multidict-6.7.1 \
    py312-openai-2.34.0_1 \
    py312-onnxruntime-1.27.0 \
    py312-opentelemetry-api-1.43.0 \
    py312-packaging-26.2 \
    py312-propcache-0.4.1_1 \
    py312-pydantic-core-2.46.4_1 \
    py312-pydantic2-2.13.4 \
    py312-pygments-2.20.0 \
    py312-python-dotenv-1.2.2 \
    py312-pyyaml-6.0.3 \
    py312-referencing-0.37.0 \
    py312-regex-2026.5.9 \
    py312-requests-2.34.2 \
    py312-rich-15.0.0 \
    py312-rpds-py-0.30.0_5 \
    py312-shellingham-1.5.4_1 \
    py312-sniffio-1.3.1 \
    py312-sqlite3-3.12.13_10 \
    py312-sqlite-vec-0.1.9 \
    py312-tiktoken-0.13.0_1 \
    py312-tokenizers-0.23.1_2 \
    py312-tqdm-4.68.3 \
    py312-typer-0.26.7 \
    py312-typing-extensions-4.15.0 \
    py312-typing-inspection-0.4.2 \
    py312-urllib3-2.7.0,1 \
    py312-yarl-1.24.2 \
    py312-zipp-3.23.0
```

Headroom uses ONNX Runtime through the ort crate's dynamic loading feature on
FreeBSD. The ONNX Runtime shared library provided by the Python environment is
resolved automatically at runtime.

---

### Step 1: Apply the temporary Cargo workspace patch

Open the root Cargo.toml of the Headroom workspace and add:

```toml
[patch.crates-io]
ort = { git = "https://github.com/pykeio/ort", branch = "main" }
```

> [!NOTE]
> This temporary patch redirects the ort dependency to the official upstream
> repository, which already contains the merged FreeBSD support introduced by
> PR #601. Once a new ort release (for example v2.0.0-rc.13 or later) is
> published on crates.io, this patch can be removed.

---

### Step 2: Apply the FreeBSD build configuration

Open `crates/headroom-core/Cargo.toml`.

Because the current ort release does not yet provide pre-built ONNX Runtime
support for x86_64-unknown-freebsd, Headroom must be configured to use
dynamic loading.

### 2.1 Update the default Unix target

Locate:

```toml
[target.'cfg(all(not(windows), not(all(target_os = "macos", target_arch = "x86_64"))))'.dependencies]
```

Replace with:

```toml
[target.'cfg(all(not(windows), not(target_os = "freebsd"), not(all(target_os = "macos", target_arch = "x86_64"))))'.dependencies]
```

### 2.2 Add the FreeBSD target

Immediately before the Windows dependency section, add:

```toml
[target.'cfg(target_os = "freebsd")'.dependencies]

# Load the ONNX Runtime shared library dynamically at runtime.
fastembed = { version = "5", default-features = false, features = [
    "hf-hub-rustls-tls",
    "ort-load-dynamic",
    "image-models",
] }

ort = { version = "2.0.0-rc.12", default-features = false, features = ["load-dynamic"] }
```

> [!TIP]
> If you are building from a branch or fork where these changes have already
> been applied, you can skip this step.

---

### Step 3: Build Headroom

Build the production proxy and wheel:

```bash
make build-proxy
make build-wheel
```

This compiles the native Rust extension in release mode with link-time
optimization (LTO) enabled.

---

### Step 4: Install the wheel

Install the generated wheel into your Python environment:

```bash
uv pip install --no-deps target/wheels/*.whl
```

---

### Step 5: Verify the installation

Run:

```bash
headroom doctor
```

The command should complete successfully and report a healthy installation.

Once a new ort release containing the merged FreeBSD support becomes available
on crates.io, the temporary workspace patch described in Step 1 can be removed.
