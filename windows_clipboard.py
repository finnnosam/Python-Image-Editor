"""Windows image clipboard support without a pywin32 dependency."""
import ctypes
from ctypes import wintypes
import io
import sys

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
