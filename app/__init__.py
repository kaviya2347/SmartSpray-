"""SmartSpray backend package."""

# Import-order guard: on Linux machines with a CUDA build of PyTorch, the
# `triton` extension segfaults if it is first loaded AFTER TensorFlow (the app
# loads TensorFlow for Module 3 and PyTorch/Ultralytics for Module 2 in the
# same process). Loading triton first avoids the crash. It is a no-op on
# macOS/Windows or when triton is not installed.
import importlib.util as _importlib_util

if _importlib_util.find_spec("triton") is not None:  # pragma: no cover
    try:
        import triton  # noqa: F401
    except Exception:
        pass
