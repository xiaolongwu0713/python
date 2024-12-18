import tkinter as tk
from tkinter import messagebox
import os
import random
import pygame
from datetime import datetime


class AudioScoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Scoring Program")
        self.audio_files = []
        self.current_audio_index = 0
        self.scores = []
        self.user_name = ""
        self.output_file = ""

        # Specify folder path directly (replace this path with your own)
        self.folder_path = "D:/data/BaiduSyncdisk/speech_Southmead/evaluation_matrix/GUI/"  # Change this to your folder

        # GUI Elements
        self.label_name = tk.Label(root, text="Enter your name:")
        self.label_name.pack(pady=5)
        self.entry_name = tk.Entry(root)
        self.entry_name.pack(pady=5)

        self.label_instruction = tk.Label(root, text="Please wait until the audio plays...")
        self.label_instruction.pack(pady=5)

        self.score_label = tk.Label(root, text="Enter score (1-5):")
        self.score_label.pack(pady=5)
        self.entry_score = tk.Entry(root, state=tk.DISABLED)
        self.entry_score.pack(pady=5)

        self.button_submit = tk.Button(root, text="Submit Score", command=self.submit_score, state=tk.DISABLED)
        self.button_submit.pack(pady=5)

        self.button_replay = tk.Button(root, text="Replay Audio", command=self.replay_audio, state=tk.DISABLED)
        self.button_replay.pack(pady=5)

        # Initialize pygame mixer
        pygame.mixer.init()

        self.load_audio_files()

    def load_audio_files(self):
        # Automatically load audio files from the specified folder
        if not os.path.exists(self.folder_path):
            messagebox.showerror("Error", f"Folder '{self.folder_path}' does not exist.")
            self.root.quit()
            return

        self.audio_files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path)
                            if f.lower().endswith(('.wav', '.mp3', '.ogg'))]

        if not self.audio_files:
            messagebox.showerror("Error", "No audio files found in the specified folder.")
            self.root.quit()
            return

        random.shuffle(self.audio_files)  # Randomize audio files
        messagebox.showinfo("Loaded", f"Loaded {len(self.audio_files)} audio files. Enter your name and start!")

    def start_session(self):
        self.user_name = self.entry_name.get().strip()
        if not self.user_name:
            messagebox.showerror("Error", "Please enter your name before proceeding.")
            return

        # Prepare output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = f"D:/data/BaiduSyncdisk/speech_Southmead/evaluation_matrix/audio_scores_GUI/{self.user_name}_{timestamp}.txt"

        self.current_audio_index = 0
        self.play_audio()

    def play_audio(self):
        if self.current_audio_index < len(self.audio_files):
            audio_file = self.audio_files[self.current_audio_index]
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            self.label_instruction.config(text=f"Now playing: {os.path.basename(audio_file)}")
            self.entry_score.config(state=tk.NORMAL)
            self.button_submit.config(state=tk.NORMAL)
            self.button_replay.config(state=tk.NORMAL)
        else:
            self.finish_session()

    def replay_audio(self):
        pygame.mixer.music.stop()
        self.play_audio()

    def submit_score(self):
        score = self.entry_score.get().strip()
        if not score.isdigit() or not (0 <= int(score) <= 10):
            messagebox.showerror("Error", "Please enter a valid score between 0 and 10.")
            return

        # Save score and move to next audio
        self.save_score_to_file(os.path.basename(self.audio_files[self.current_audio_index]), score)
        self.current_audio_index += 1
        self.entry_score.delete(0, tk.END)
        self.play_audio()

    def save_score_to_file(self, audio_file, score):
        with open(self.output_file, 'a') as f:
            f.write(f"{audio_file}, {score}\n")

    def finish_session(self):
        pygame.mixer.music.stop()
        messagebox.showinfo("Finished", f"All audio files have been scored! Scores saved to {self.output_file}")
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioScoringApp(root)
    start_button = tk.Button(root, text="Start", command=app.start_session)
    start_button.pack(pady=5)
    root.mainloop()
