# scene_square_to_circle.py
from manim import *

class SquareToCircle(Scene):
    def construct(self):
        s = Square().set_fill(BLUE, 0.5).set_stroke(WHITE, 2)
        self.play(Create(s))
        self.play(Transform(s, Circle()))
        self.wait(0.2)
