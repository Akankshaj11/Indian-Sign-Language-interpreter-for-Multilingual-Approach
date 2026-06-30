


import cv2
import time

from realtime_static_prediction import StaticPredictor
from realtime_dynamic_prediction import DynamicPredictor

from translation_manager import TranslationManager
from text_to_speech import TextToSpeech
from ui_manager import UIManager

# ==========================================
# LOAD MODULES
# ==========================================

print("=" * 60)
print("Hybrid Sign Language Interpreter")
print("=" * 60)

static_predictor = StaticPredictor()
dynamic_predictor = DynamicPredictor()

translator = TranslationManager()

tts = TextToSpeech()

ui = UIManager()

print("All Modules Loaded Successfully!")

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise Exception("Unable to open webcam.")

# ==========================================
# APPLICATION STATE
# ==========================================

mode = "dynamic"

language_name = "English"
language_code = "en"

sentence = []

prediction = "Show Gesture"

translated_prediction = ""

translated_sentence = ""

confidence = 0

# ==========================================
# LANGUAGE MAP
# ==========================================

LANGUAGES = {

    ord("1"): ("English", "en"),

    ord("2"): ("Hindi", "hi"),

    ord("3"): ("Marathi", "mr")

}

# ==========================================
# FPS
# ==========================================

previous_time = time.time()

fps = 0

# ==========================================
# WINDOW
# ==========================================

WINDOW_NAME = "Hybrid Sign Language Interpreter"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:

        break

    frame = cv2.flip(frame, 1)

    # --------------------------------------

    current_time = time.time()

    dt = current_time - previous_time

    if dt != 0:

        fps = int(1 / dt)

    previous_time = current_time

    # --------------------------------------
    # Prediction
    # --------------------------------------

    if mode == "dynamic":

        prediction, confidence, frame = \
            dynamic_predictor.predict(frame)

    else:

        prediction, confidence, frame = \
            static_predictor.predict(frame)

    # --------------------------------------
    # Translation
    # --------------------------------------

    translated_prediction = translator.translate_prediction(

        prediction,

        language_code

    )

    translated_sentence = translator.translate_sentence(

        sentence,

        language_code

    )
        # ======================================
    # DRAW USER INTERFACE
    # ======================================

    frame = ui.draw(

        frame=frame,

        mode=mode,

        language=language_name,

        fps=fps,

        prediction=prediction,

        confidence=confidence,

        translated_prediction=translated_prediction,

        sentence=" ".join(sentence),

        translated_sentence=translated_sentence

    )

    # ======================================
    # SHOW WINDOW
    # ======================================

    cv2.imshow(

        WINDOW_NAME,

        frame

    )

    # ======================================
    # KEYBOARD
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------
    # STATIC MODE
    # --------------------------------------

    if key == ord("s"):

        mode = "static"

        print("Mode -> STATIC")

        continue

    # --------------------------------------
    # DYNAMIC MODE
    # --------------------------------------

    if key == ord("d"):

        mode = "dynamic"

        print("Mode -> DYNAMIC")

        continue

    # --------------------------------------
    # LANGUAGE
    # --------------------------------------

    if key in LANGUAGES:

        language_name, language_code = LANGUAGES[key]

        print("Language ->", language_name)

        continue

    # ======================================
    # SENTENCE BUILDER
    # ======================================

    if key == 32:

        # SPACE

        invalid_predictions = [

            "",

            "Show Gesture",

            "No Hand",

            "No Hand Detected",

            "Uncertain"

        ]

        if prediction not in invalid_predictions:

            sentence.append(prediction)

            translated_sentence = translator.translate_sentence(

                sentence,

                language_code

            )

            print("Added :", prediction)

        continue

    # ======================================
    # DELETE LAST WORD
    # ======================================

    if key == 8:

        if len(sentence):

            removed = sentence.pop()

            translated_sentence = translator.translate_sentence(

                sentence,

                language_code

            )

            print("Deleted :", removed)

        continue

    # ======================================
    # CLEAR SENTENCE
    # ======================================

    if key == ord("c"):

        sentence.clear()

        translated_sentence = ""

        print("Sentence Cleared")

        continue

        # ======================================
    # TEXT TO SPEECH
    # ======================================

    if key == ord("p"):

        text = translated_sentence.strip()

        if text == "":
            text = translated_prediction.strip()

        if text != "":

            try:

                tts.speak(
                    text,
                    language_code
                )

            except Exception as e:

                print("TTS Error:", e)

        continue

    # ======================================
    # STOP SPEAKING
    # ======================================

    if key == ord("x"):

        try:

            tts.stop()

        except:
            pass

        continue

    # ======================================
    # EXIT
    # ======================================

    if key == 27:

        print("Closing application...")

        break


# ==========================================
# CLEANUP
# ==========================================

print("Releasing resources...")

cap.release()

try:
    static_predictor.close()
except:
    pass

try:
    dynamic_predictor.close()
except:
    pass

try:
    tts.stop()
except:
    pass

cv2.destroyAllWindows()

print("Application Closed Successfully.")