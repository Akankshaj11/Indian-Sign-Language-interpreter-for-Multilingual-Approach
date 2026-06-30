import cv2
from matplotlib.pyplot import draw
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


class UIManager:

    def __init__(self):

        # =====================================
        # COLORS
        # =====================================

        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)

        self.BLUE = (255, 120, 0)
        self.YELLOW = (0, 255, 255)

        self.CYAN = (255, 255, 0)

        # =====================================
        # FONTS
        # =====================================

       

        font_candidates = [
            r"C:\Windows\Fonts\Nirmala.ttc",
            r"C:\Windows\Fonts\Mangal.ttf",
            r"C:\Windows\Fonts\Aparaj.ttf",
            r"C:\Windows\Fonts\Kokila.ttf"
        ]

        font_path = None

        for path in font_candidates:
            if os.path.exists(path):
                font_path = path
                break

        # This must be OUTSIDE the for loop
        if font_path:

            print("Using font:", font_path)

            self.font_large = ImageFont.truetype(font_path, 28)
            self.font_medium = ImageFont.truetype(font_path, 22)
            self.font_small = ImageFont.truetype(font_path, 18)

        else:

            print("Unicode font not found!")

            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()



    # =====================================
    # Transparent Rectangle
    # =====================================

    def transparent_box(
            self,
            frame,
            x1,
            y1,
            x2,
            y2,
            alpha=0.45
    ):

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            self.BLACK,
            -1
        )

        cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1-alpha,
            0,
            frame
        )

    # =====================================
    # PIL DRAW
    # =====================================

    def get_draw(self, frame):

        image = Image.fromarray(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
        )

        draw = ImageDraw.Draw(image)

        return image, draw

    # =====================================
    # Convert Back
    # =====================================

    def to_cv(self, image):

        return cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )

    # =====================================
    # Draw Top Banner
    # =====================================

    

    def draw_top_banner(

            self,

            frame,

            mode,

            language,

            fps,

            prediction,

            confidence,

            translated_prediction

    ):
        
        h, w = frame.shape[:2]

        self.transparent_box(
            frame,
            0,
            0,
            w,
            90
        )

        image, draw = self.get_draw(frame)

        draw.text(
            (20,10),
            f"MODE : {mode.upper()}",
            font=self.font_medium,
            fill=(255,255,0)
        )

        draw.text(
            (320,10),
            f"LANGUAGE : {language.upper()}",
            font=self.font_medium,
            fill=(0,255,255)
        )

        draw.text(
            (650,10),
            f"FPS : {fps}",
            font=self.font_medium,
            fill=(0,255,0)
        )

        draw.text(
            (20,50),
            f"Prediction : {prediction}",
            font=self.font_medium,
            fill=(0,255,0)
        )

        draw.text(
            (20,85),
            f"Confidence : {confidence:.1f} %",
            font=self.font_small,
            fill=(255,255,255)
        )

        # draw.text(
        #     (450,50),
        #     f"Translation : {translated_prediction}",
        #     font=self.font_medium,
        #     fill=(255,255,255)
        # )


        # Limit translation length
       # Limit translation length
        top_translation = translated_prediction

        if len(top_translation) > 18:
            top_translation = top_translation[:15] + "..."

        draw.text(
            (430,50),
            f"Translation : {top_translation}",
            font=self.font_medium,
            fill=(255,255,255)
        )



        return self.to_cv(image)
        # =====================================
    # Draw Bottom Banner
    # =====================================

    def draw_bottom_banner(

            self,

            frame,

            sentence,

            translated_sentence

    ):

        h, w = frame.shape[:2]

        self.transparent_box(
            frame,
            0,
            h-130,
            w,
            h
        )

        image, draw = self.get_draw(frame)

        draw.text(
            (20, h-120),
            "Sentence :",
            font=self.font_medium,
            fill=(255,255,0)
        )

     
        display_sentence = sentence

        if len(display_sentence) > 35:
            display_sentence = display_sentence[:32] + "..."

        draw.text(
            (170, h-120),
            display_sentence,
            font=self.font_medium,
            fill=(255,255,255)
        )

        draw.text(
            (20, h-80),
            "Translation :",
            font=self.font_medium,
            fill=(0,255,255)
        )

     

        display_translation = translated_sentence

        if len(display_translation) > 35:
            display_translation = display_translation[:32] + "..."

        draw.text(
            (170, h-80),
            display_translation,
            font=self.font_medium,
            fill=(255,255,255)
        )




        return self.to_cv(image)

    # =====================================
    # Draw Hotkeys
    # =====================================

    def draw_hotkeys(self, frame):

        h, w = frame.shape[:2]

        image, draw = self.get_draw(frame)


        draw.text(
            (20, h-42),
            "S Static   D Dynamic   1 English   2 Hindi   3 Marathi",
            font=self.font_small,
            fill=(255,255,255)
        )

        draw.text(
            (20, h-20),
            "SPACE Add   BKSP Delete   C Clear   P Speak   ESC Exit",
            font=self.font_small,
            fill=(255,255,255)
        )



        return self.to_cv(image)

    # =====================================
    # Main Draw Function
    # =====================================

    def draw(

            self,

            frame,

            mode,

            language,

            fps,

            prediction,

            confidence,

            translated_prediction,

            sentence,

            translated_sentence

    ):

        frame = self.draw_top_banner(

            frame,

            mode,

            language,

            fps,

            prediction,

            confidence,

            translated_prediction

        )

        frame = self.draw_bottom_banner(

            frame,

            sentence,

            translated_sentence

        )

        frame = self.draw_hotkeys(frame)

        return frame