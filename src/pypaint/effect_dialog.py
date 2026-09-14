"""Descriptor-driven effect dialog; widgets and publication remain on the UI thread."""

import tkinter as tk
from tkinter import messagebox

from pypaint.editor import apply, compute
from pypaint.jobs import Job
from pypaint.state import context_for


def show(window, operation):
    window._finish_clipboard_edit()
    window._finish_raster_stroke()
    document = context_for(window).document
    layer = document.layers[document.active_layer]
    if not layer.is_raster:
        messagebox.showinfo(operation.name, "Select a raster layer to use this operation.")
        return
    source = layer.image.snapshot()
    selection = window.selection_mask.copy() if window._selection_pixel_box() is not None else None
    if selection is not None and hasattr(selection, "snapshot"):
        selection = selection.snapshot()
    generation = document.generation
    dialog = tk.Toplevel(window.root)
    dialog.title(operation.name)
    dialog.transient(window.root)
    variables = {}
    for parameter in operation.parameters:
        variable = tk.DoubleVar(dialog, value=parameter.default)
        variables[parameter.name] = variable
        tk.Label(dialog, text=parameter.name.title()).pack()
        tk.Scale(
            dialog,
            from_=parameter.minimum,
            to=parameter.maximum,
            resolution=0.5,
            orient="horizontal",
            variable=variable,
            length=240,
        ).pack()
    status = tk.StringVar(dialog, value="Preview ready to calculate")
    tk.Label(dialog, textvariable=status).pack(padx=12, pady=6)
    state = {"job": None, "surface": None, "closed": False}

    def close():
        state["closed"] = True
        if state["job"]:
            state["job"].cancellation.cancel()
            window.job_callbacks.pop(state["job"].operation_id, None)
        window._effect_preview = None
        window.request_redraw()
        dialog.destroy()

    def preview():
        if state["job"]:
            state["job"].cancellation.cancel()
        values = {key: variable.get() for key, variable in variables.items()}
        job = Job(
            document.id, generation, document.state_id, preview_key="effect-preview", priority=0
        )
        state["job"] = job
        state["surface"] = None
        apply_button.configure(state="disabled")
        status.set("Calculating…")

        def publish(result):
            if state["closed"]:
                return
            if result.error:
                status.set(str(result.error))
                return
            state["surface"] = result.value
            window._effect_preview = (document.id, layer.id, result.value)
            window.request_redraw()
            apply_button.configure(state="normal")
            status.set("Preview")

        try:
            window.jobs.submit(
                job,
                lambda token: compute(source, selection, operation, values, token),
                32 * 1024**2,
            )
            window.job_callbacks[job.operation_id] = publish
        except Exception as error:
            status.set(str(error))

    def commit():
        try:
            apply(document, layer.id, generation, state["surface"], operation.name)
            window.documents[document.id]["modified"] = document.modified
            window.notify_globe_document_changed()
            window._highlight_document_tabs()
            close()
        except Exception as error:
            status.set(str(error))

    tk.Button(dialog, text="Preview", command=preview).pack(side="left", padx=6, pady=8)
    apply_button = tk.Button(dialog, text="Apply", command=commit, state="disabled")
    apply_button.pack(side="left", padx=6, pady=8)
    tk.Button(dialog, text="Cancel", command=close).pack(side="left", padx=6, pady=8)
    dialog.protocol("WM_DELETE_WINDOW", close)
    dialog.bind("<Escape>", lambda event: close())
    dialog.grab_set()
    preview()
