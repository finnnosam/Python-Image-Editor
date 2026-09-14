"""Windows clipboard, shortcut parsing, resources and user configuration."""

import ctypes
import io
import os
import re
import sys
from ctypes import wintypes
from importlib.resources import files
from pathlib import Path

from PIL import Image, ImageGrab


def _api():
    if sys.platform != "win32":
        raise OSError("Image clipboard support requires Windows.")
    user = ctypes.WinDLL("user32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = [
        (user.OpenClipboard, [wintypes.HWND], wintypes.BOOL),
        (user.CloseClipboard, [], wintypes.BOOL),
        (user.EmptyClipboard, [], wintypes.BOOL),
        (user.RegisterClipboardFormatW, [wintypes.LPCWSTR], wintypes.UINT),
        (user.SetClipboardData, [wintypes.UINT, wintypes.HANDLE], wintypes.HANDLE),
        (user.GetClipboardData, [wintypes.UINT], wintypes.HANDLE),
        (kernel.GlobalAlloc, [wintypes.UINT, ctypes.c_size_t], wintypes.HGLOBAL),
        (kernel.GlobalLock, [wintypes.HGLOBAL], ctypes.c_void_p),
        (kernel.GlobalUnlock, [wintypes.HGLOBAL], wintypes.BOOL),
        (kernel.GlobalSize, [wintypes.HGLOBAL], ctypes.c_size_t),
        (kernel.GlobalFree, [wintypes.HGLOBAL], wintypes.HGLOBAL),
    ]
    for function, arguments, result in signatures:
        function.argtypes = arguments
        function.restype = result
    return user, kernel


def copy_image(image, owner):
    """Publish lossless PNG plus a conventional DIB for other Windows apps."""
    user, kernel = _api()
    png = io.BytesIO()
    image.save(png, "PNG")
    bmp = io.BytesIO()
    # Legacy DIB readers ignore alpha; give them a useful white background.
    opaque = Image.new("RGB", image.size, "white")
    rgba = image.convert("RGBA")
    opaque.paste(rgba, mask=rgba.getchannel("A"))
    opaque.save(bmp, "BMP")
    png_format = user.RegisterClipboardFormatW("PNG")
    if not png_format:
        raise ctypes.WinError(ctypes.get_last_error())
    handles = []
    opened = False
    try:
        for format_id, data in ((png_format, png.getvalue()), (8, bmp.getvalue()[14:])):
            handle = kernel.GlobalAlloc(0x0002, len(data))
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            handles.append([format_id, handle])
            pointer = kernel.GlobalLock(handle)
            if not pointer:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                ctypes.memmove(pointer, data, len(data))
            finally:
                kernel.GlobalUnlock(handle)
        if not user.OpenClipboard(owner):
            raise OSError("The Windows clipboard is busy. Please try again.")
        opened = True
        if not user.EmptyClipboard():
            raise ctypes.WinError(ctypes.get_last_error())
        for entry in handles:
            if not user.SetClipboardData(*entry):
                raise ctypes.WinError(ctypes.get_last_error())
            entry[1] = None  # Windows now owns this allocation.
    finally:
        if opened:
            user.CloseClipboard()
        for _, handle in handles:
            if handle:
                kernel.GlobalFree(handle)


def paste_image():
    """Prefer PNG to preserve transparency; accept Windows bitmap images too."""
    user, kernel = _api()
    png_format = user.RegisterClipboardFormatW("PNG")
    if not png_format:
        raise ctypes.WinError(ctypes.get_last_error())
    if not user.OpenClipboard(None):
        raise OSError("The Windows clipboard is busy. Please try again.")
    data = None
    try:
        handle = user.GetClipboardData(png_format)
        if handle:
            pointer = kernel.GlobalLock(handle)
            if not pointer:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                data = ctypes.string_at(pointer, kernel.GlobalSize(handle))
            finally:
                kernel.GlobalUnlock(handle)
    finally:
        user.CloseClipboard()
    if data is not None:
        with Image.open(io.BytesIO(data)) as source:
            return source.convert("RGBA")
    result = ImageGrab.grabclipboard()
    if isinstance(result, Image.Image):
        return result.convert("RGBA")
    return None


# Read editable keyboard bindings; shortcut files never execute code.


def key_sequence(value):
    """Translate a readable shortcut (Ctrl+S, Shift+F2) to a Tk key event."""
    parts = [part.strip() for part in value.split("+")]
    modifiers = []
    aliases = {"ctrl": "Control", "control": "Control", "shift": "Shift", "alt": "Alt"}
    for part in parts[:-1]:
        modifier = aliases.get(part.lower())
        if modifier is None or modifier in modifiers:
            raise ValueError(f"Invalid modifier in {value!r}")
        modifiers.append(modifier)
    key = parts[-1]
    names = {
        "plus": "plus",
        "minus": "minus",
        "equals": "equal",
        "space": "space",
        "enter": "Return",
        "return": "Return",
        "escape": "Escape",
        "esc": "Escape",
        "tab": "Tab",
        "delete": "Delete",
        "backspace": "BackSpace",
        "up": "Up",
        "down": "Down",
        "left": "Left",
        "right": "Right",
        "home": "Home",
        "end": "End",
        "pageup": "Prior",
        "pagedown": "Next",
        "numpadplus": "KP_Add",
        "numpadminus": "KP_Subtract",
    }
    if len(key) == 1 and key.isascii() and key.isalnum():
        key = key.upper() if "Shift" in modifiers else key.lower()
    elif key.lower() in names:
        key = names[key.lower()]
    elif re.fullmatch(r"[fF]([1-9]|[12][0-9]|3[0-5])", key):
        key = key.upper()
    else:
        raise ValueError(f"Unknown key {key!r}; use letters, F1–F35, or a named key")
    modifiers.sort(key=["Control", "Alt", "Shift"].index)
    return "<" + "-".join(modifiers + ["KeyPress", key]) + ">"


def shortcut_label(sequence):
    """Convert a normalized Tk key sequence back to a compact UI label."""
    parts = sequence.removeprefix("<").removesuffix(">").split("-")
    parts = [part for part in parts if part != "KeyPress"]
    aliases = {
        "Control": "Ctrl",
        "equal": "=",
        "plus": "+",
        "minus": "-",
        "Return": "Enter",
        "BackSpace": "Backspace",
        "Prior": "PageUp",
        "Next": "PageDown",
        "KP_Add": "Numpad+",
        "KP_Subtract": "Numpad-",
    }
    return "+".join(aliases.get(part, part.upper() if len(part) == 1 else part) for part in parts)


def read_shortcuts(path, actions):
    """Return valid action/sequence pairs and readable errors for invalid lines."""
    bindings, errors, used = [], [], {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        action, separator, values = line.partition("=")
        action = action.strip()
        if not separator or action not in actions:
            errors.append(f"Line {number}: unknown action or missing '=': {action}")
            continue
        for value in values.split(","):
            if not value.strip():
                continue
            try:
                sequence = key_sequence(value.strip())
                if sequence in used:
                    raise ValueError(f"{value.strip()} is already assigned to {used[sequence]}")
            except ValueError as error:
                errors.append(f"Line {number}: {error}")
                continue
            used[sequence] = action
            bindings.append((action, sequence))
    return bindings, errors


# Installed resources and writable user configuration, independent of cwd.


def resource(name):
    return files("pypaint").joinpath("resources", name)


def shortcuts_path():
    directory = Path(
        os.environ.get("PYPAINT_CONFIG_HOME")
        or Path(os.environ.get("APPDATA", Path.home() / ".config")) / "PyPaint"
    )
    destination = directory / "shortcuts.txt"
    if destination.exists():
        return destination
    directory.mkdir(parents=True, exist_ok=True)
    legacy = Path(__file__).resolve().parents[2] / "shortcuts.txt"
    data = legacy.read_bytes() if legacy.is_file() else resource("shortcuts.txt").read_bytes()
    # Exclusive creation preserves a concurrently created override.
    try:
        with destination.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        pass
    return destination
