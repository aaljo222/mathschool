import streamlit as st
from manim import Scene, Square, Circle, BLUE, WHITE, Create, Transform
from utils.manim_runner import render_manim

class SquareToCircle(Scene):
    def construct(self):
        s = Square().set_fill(BLUE, 0.5).set_stroke(WHITE, 2)
        self.play(Create(s))
        self.play(Transform(s, Circle()))
        self.wait(0.2)

st.subheader("Manim 데모(경량 렌더)")
if st.button("렌더"):
    path = render_manim(SquareToCircle, cache_key="square_to_circle")
    st.video(str(path))
