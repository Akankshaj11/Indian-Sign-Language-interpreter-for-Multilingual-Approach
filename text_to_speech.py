import os
import tempfile
import threading

from gtts import gTTS
import pygame


class TextToSpeech:

    def __init__(self):

        pygame.mixer.init()

        self.thread = None

        self.is_speaking = False

    # ======================================
    # Internal Function
    # ======================================

    def _speak(self, text, language):

        try:

            self.is_speaking = True

            tts = gTTS(
                text=text,
                lang=language,
                slow=False
            )

            temp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp3"
            )

            temp.close()

            tts.save(temp.name)

            pygame.mixer.music.load(temp.name)

            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():

                pygame.time.Clock().tick(10)

            pygame.mixer.music.unload()

            os.remove(temp.name)

        except Exception as e:

            print("TTS Error:", e)

        self.is_speaking = False

    # ======================================
    # Public Function
    # ======================================

    def speak(self, text, language="en"):

        if text == "":
            return

        if self.is_speaking:
            return

        self.thread = threading.Thread(

            target=self._speak,

            args=(text, language),

            daemon=True

        )

        self.thread.start()

    # ======================================
    # Stop Speaking
    # ======================================

    def stop(self):

        if pygame.mixer.music.get_busy():

            pygame.mixer.music.stop()

        self.is_speaking = False