from googletrans import Translator

# ==========================================
# Dataset Label Mapping
# ==========================================

WORD_MAP = {
    "thank_you": "thank you",
    "bye": "goodbye",
    "hello": "hello",
    "yes": "yes",
    "namaste": "namaste",
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "bad": "bad",
    "good_morning": "good morning",
    "good_afternoon": "good afternoon",
    "good_evening": "good evening",
    "namaste": "namaste",
    "kind": "kind",
    "me": "me",
    "you": "you",
    "please": "please",
    "sorry": "sorry",
    "scared": "scared",
    "tired": "tired",
    "understand": "understand",

}

# ==========================================
# Custom Offline Translations
# ==========================================

CUSTOM_TRANSLATIONS = {

    "en": {
        "hello": "hello",
        "goodbye": "goodbye",
        "thank you": "thank you",
        "yes": "yes",
        "namaste": "namaste"
    },

    "hi": {
        "hello": "नमस्ते",
        "goodbye": "अलविदा",
        "thank you": "धन्यवाद",
        "yes": "हाँ",
        "namaste": "नमस्ते"
    },

    "mr": {
        "hello": "नमस्कार",
        "goodbye": "निरोप",
        "thank you": "धन्यवाद",
        "yes": "हो",
        "namaste": "नमस्कार"
    }

}


class TranslationManager:

    def __init__(self):

        self.translator = Translator()

        self.last_prediction = None
        self.last_sentence = None

        self.cached_prediction = ""
        self.cached_sentence = ""

    # ======================================
    # Translate Single Prediction
    # ======================================

    def translate_prediction(self, prediction, lang):

        if prediction in ["", "No Hand", "Show Gesture", "Uncertain"]:
            return prediction

        key = (prediction, lang)

        if key == self.last_prediction:
            return self.cached_prediction

        english = WORD_MAP.get(
            prediction,
            prediction.replace("_", " ")
        )

        if lang == "en":
            translated = english

        else:

            try:

                translated = self.translator.translate(
                    english,
                    dest=lang
                ).text

                print("Prediction Translation:", translated)

            except Exception:

                translated = CUSTOM_TRANSLATIONS.get(
                    lang,
                    {}
                ).get(
                    english.lower(),
                    english
                )

        self.cached_prediction = translated
        self.last_prediction = key

        return translated

    # ======================================
    # Translate Sentence
    # ======================================

    def translate_sentence(self, words, lang):

        if not words:
            return ""

        key = (tuple(words), lang)

        if key == self.last_sentence:
            return self.cached_sentence

        english = " ".join(
            WORD_MAP.get(
                word,
                word.replace("_", " ")
            )
            for word in words
        )

        if lang == "en":

            translated = english

        else:

            try:

                translated = self.translator.translate(
                    english,
                    dest=lang
                ).text

                print("Sentence Translation:", translated)

            except Exception:

                translated = english

        self.cached_sentence = translated
        self.last_sentence = key

        return translated
    

    