"""Read editable keyboard bindings; shortcut files never execute code."""
import re


def key_sequence(value):
    """Translate a readable shortcut (Ctrl+S, Shift+F2) to a Tk key event."""
    parts = [part.strip() for part in value.split("+")]
    modifiers = []
    aliases = {"ctrl": "Control", "control": "Control",
               "shift": "Shift", "alt": "Alt"}
    for part in parts[:-1]:
        modifier = aliases.get(part.lower())
        if modifier is None or modifier in modifiers:
            raise ValueError(f"Invalid modifier in {value!r}")
        modifiers.append(modifier)
    key = parts[-1]
    names = {"plus": "plus", "minus": "minus", "equals": "equal",
             "space": "space", "enter": "Return", "return": "Return",
             "escape": "Escape", "esc": "Escape", "tab": "Tab",
             "delete": "Delete", "backspace": "BackSpace",
             "up": "Up", "down": "Down", "left": "Left", "right": "Right",
             "home": "Home", "end": "End", "pageup": "Prior",
             "pagedown": "Next", "numpadplus": "KP_Add",
             "numpadminus": "KP_Subtract"}
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
