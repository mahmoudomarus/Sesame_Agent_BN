import customtkinter as ctk
import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
import logging
import yaml
import os
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class VoiceAgentApp(ctk.CTk):
    def __init__(self, config: dict, start_callback: Callable, stop_callback: Callable, interrupt_callback: Callable):
        super().__init__()

        self.config = config
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.interrupt_callback = interrupt_callback

        self.is_running = False

        # --- Window Setup ---
        self.title("Bignoodle Voice Agent")
        gui_config = config.get('gui', {})
        width = gui_config.get('window_width', 800)
        height = gui_config.get('window_height', 600)
        self.geometry(f"{width}x{height}")
        ctk.set_appearance_mode(gui_config.get('theme', 'dark').lower()) # dark, light, system
        ctk.set_default_color_theme("blue") # Or green, dark-blue

        # --- Grid Layout ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0) # Column for controls
        self.grid_rowconfigure(0, weight=1) # Row for conversation log
        self.grid_rowconfigure(1, weight=0) # Row for status
        self.grid_rowconfigure(2, weight=0) # Row for controls

        # --- Widgets ---

        # Conversation Log (Row 0, Col 0)
        self.log_textbox = scrolledtext.ScrolledText(self, wrap=tk.WORD, state='disabled', height=15, bg="#2b2b2b", fg="#d3d3d3", font=("Arial", 11))
        self.log_textbox.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew", columnspan=2)

        # Status Bar (Row 1, Col 0)
        self.status_label = ctk.CTkLabel(self, text="Status: Idle", anchor="w", font=("Arial", 12))
        self.status_label.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Control Frame (Row 2, Col 0+1)
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=(5, 10), sticky="ew")
        self.control_frame.grid_columnconfigure((0, 1, 2), weight=1) # Distribute controls

        # Start/Stop Button
        self.start_stop_button = ctk.CTkButton(self.control_frame, text="Start Agent", command=self.toggle_agent)
        self.start_stop_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Interrupt Button
        self.interrupt_button = ctk.CTkButton(self.control_frame, text="Interrupt", command=self.interrupt_agent, state="disabled")
        self.interrupt_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Placeholder for Emotion Sliders (if enabled in config)
        if gui_config.get('show_emotion_controls', False):
            self.emotion_frame = ctk.CTkFrame(self.control_frame)
            self.emotion_frame.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
            self.emotion_frame.grid_columnconfigure(0, weight=1)
            # Add sliders or other controls here later
            self.emotion_label = ctk.CTkLabel(self.emotion_frame, text="Emotion (NYI):", anchor="w")
            self.emotion_label.grid(row=0, column=0, padx=5, pady=2)
            # Example slider placeholder
            # self.humor_slider = ctk.CTkSlider(self.emotion_frame, from_=0, to=1, command=self.update_emotion)
            # self.humor_slider.grid(row=1, column=0, padx=5, pady=2, sticky="ew")
            # self.humor_slider.set(0.5)

        # --- Queue for GUI Updates from Backend Threads ---
        self.gui_queue = queue.Queue()
        self.after(100, self.process_gui_queue) # Start checking the queue periodically

    def toggle_agent(self):
        if not self.is_running:
            logger.info("GUI: Start button clicked.")
            self.start_stop_button.configure(text="Stop Agent")
            self.interrupt_button.configure(state="normal")
            self.update_status("Starting...")
            self.is_running = True
            # Trigger the actual start logic via callback
            self.start_callback()
        else:
            logger.info("GUI: Stop button clicked.")
            self.start_stop_button.configure(text="Start Agent")
            self.interrupt_button.configure(state="disabled")
            self.update_status("Stopping...")
            self.is_running = False
            # Trigger the actual stop logic via callback
            self.stop_callback()
            self.update_status("Idle") # Set status back to Idle after stop signal sent

    def interrupt_agent(self):
        if self.is_running:
            logger.info("GUI: Interrupt button clicked.")
            self.update_status("Interrupting...")
            # Trigger the actual interrupt logic via callback
            self.interrupt_callback()
            # Status might be updated again by backend after interrupt handling

    def update_status(self, message: str):
        """Thread-safe method to update the status label."""
        self.gui_queue.put(("status", message))

    def add_log_message(self, role: str, message: str):
        """Thread-safe method to add a message to the conversation log."""
        self.gui_queue.put(("log", (role, message)))

    def _update_status_label(self, message: str):
        self.status_label.configure(text=f"Status: {message}")

    def _add_to_log(self, role: str, message: str):
        self.log_textbox.configure(state='normal')
        timestamp = time.strftime("%H:%M:%S")
        prefix = f"[{timestamp}] {role.capitalize()}: "
        self.log_textbox.insert(tk.END, prefix + message + "\n\n")
        self.log_textbox.configure(state='disabled')
        self.log_textbox.see(tk.END) # Auto-scroll

    def process_gui_queue(self):
        """Process updates from the backend thread queue."""
        try:
            while True: # Process all messages currently in the queue
                msg_type, data = self.gui_queue.get_nowait()
                if msg_type == "status":
                    self._update_status_label(data)
                elif msg_type == "log":
                    role, message = data
                    self._add_to_log(role, message)
                else:
                    logger.warning(f"Unknown GUI queue message type: {msg_type}")
        except queue.Empty:
            pass # No more messages
        finally:
            # Reschedule the queue check
            self.after(100, self.process_gui_queue)

    def update_emotion(self, value):
         # Placeholder for when emotion sliders are implemented
         # This would likely update a shared state or send a message
         # to the backend thread.
         logger.debug(f"Emotion slider value changed (NYI): {value}")
         pass

    def on_closing(self):
        """Handle window closing event."""
        logger.info("GUI closing requested.")
        if self.is_running:
            self.stop_callback() # Ensure backend is signalled to stop
        self.destroy()

# --- Placeholder Callbacks (to be replaced by main.py integration) ---
def placeholder_start():
    print("Placeholder START function called.")
    # Simulate backend starting and sending updates
    threading.Timer(1.0, lambda: app.update_status("Listening...")).start()
    threading.Timer(2.0, lambda: app.add_log_message("user", "Hello there agent.")).start()
    threading.Timer(3.0, lambda: app.update_status("Thinking...")).start()
    threading.Timer(4.5, lambda: app.add_log_message("assistant", "Hi user! I am thinking.")).start()
    threading.Timer(5.5, lambda: app.update_status("Speaking...")).start()
    threading.Timer(7.0, lambda: app.update_status("Listening...")).start()

def placeholder_stop():
    print("Placeholder STOP function called.")

def placeholder_interrupt():
    print("Placeholder INTERRUPT function called.")
    app.update_status("Interrupted! Listening...")

if __name__ == '__main__':
    # Load config needed for GUI settings
    CONFIG_PATH = os.getenv("CONFIG_PATH", "../config/config.yaml") # Adjust path if running from src/
    config = {}
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded for GUI: {CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Error loading configuration for GUI: {e}. Using defaults.")

    # Set up basic logging for the GUI itself
    log_level = config.get('log_level', 'INFO').upper()
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    app = VoiceAgentApp(
        config=config,
        start_callback=placeholder_start,
        stop_callback=placeholder_stop,
        interrupt_callback=placeholder_interrupt
    )
    app.protocol("WM_DELETE_WINDOW", app.on_closing) # Handle window close button
    app.mainloop() 