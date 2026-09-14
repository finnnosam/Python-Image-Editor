"""Desktop composition root; importing the package does not create Tk objects."""


def main():
    import tkinter as tk

    from pypaint.window import PaintApp

    root = tk.Tk()
    try:
        root.state("zoomed")
    except tk.TclError:
        try:
            root.attributes("-zoomed", True)
        except tk.TclError:
            pass
    app = PaintApp(root)

    def report_error(kind, error, traceback):
        import logging
        from tkinter import messagebox

        from pypaint.history import history_for
        from pypaint.state import context_for

        logging.getLogger("pypaint").error("UI operation failed", exc_info=(kind, error, traceback))
        if isinstance(error, (OSError, MemoryError)):
            document = context_for(app).document
            history_for(document).cancel(document)
            app.cancel_gesture()
            messagebox.showerror(
                "Operation could not finish",
                f"{error}\nThe last committed document is retained. Free disk space or increase the history budget, then retry.",
            )
        else:
            messagebox.showerror("Operation failed", str(error))

    root.report_callback_exception = report_error
    root.mainloop()
