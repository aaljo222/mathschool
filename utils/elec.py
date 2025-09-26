# utils/elec.py
import numpy as np, math
import plotly.graph_objects as go

def ohm_dc_result(V=None, I=None, R=None):
    if V is None: V = I*R
    if I is None: I = V/R if R!=0 else float("inf")
    if R is None: R = V/I if I!=0 else float("inf")
    P = V*I
    return V, I, R, P

def series_parallel_req(values):
    vals = [float(v) for v in values if float(v) > 0]
    if not vals: return 0.0, 0.0
    r_series = float(np.sum(vals))
    r_parallel = 1.0 / np.sum([1.0/v for v in vals]) if all(v>0 for v in vals) else float("inf")
    return r_series, r_parallel

def phasor_fig(V, I, phi, title="Phasor"):
    th = np.linspace(0, 2*np.pi, 360)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines",
                             name="unit circle", opacity=0.35))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 0], mode="lines+markers", name="V(참조)"))
    fig.add_trace(go.Scatter(x=[0, math.cos(phi)], y=[0, math.sin(phi)],
                             mode="lines+markers", name="I(위상 이동)"))
    lim = 1.3
    fig.update_xaxes(range=[-lim, lim], zeroline=True)
    fig.update_yaxes(range=[-lim, lim], scaleanchor="x", scaleratio=1)
    fig.update_layout(template="plotly_white", title=title, height=420)
    return fig

def waveform_fig(Vp, Ip, f, phi, dur=0.1):
    t = np.linspace(0, dur, 1000)
    v = Vp*np.sin(2*np.pi*f*t)
    i = Ip*np.sin(2*np.pi*f*t + phi)
    p = v*i
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=v, name="v(t)"))
    fig.add_trace(go.Scatter(x=t, y=i, name="i(t)"))
    fig.add_trace(go.Scatter(x=t, y=p, name="p(t)=v·i", opacity=0.5))
    fig.update_layout(template="plotly_white", height=420,
                      title="시간영역 파형", xaxis_title="t (s)")
    return fig

def make_parallel_animation(Rs, V, seconds=5, fps=30):
    Rs = [float(r) for r in Rs if float(r) > 0]
    N = len(Rs)
    fig = go.Figure()
    if N == 0: return fig

    I_each = [V/r for r in Rs]
    I_tot  = float(sum(I_each))
    Req    = 1.0 / float(sum(1.0/r for r in Rs))

    xL, xR = 1.0, 9.0
    x1, x2 = 2.0, 8.0
    ys     = np.linspace(N-1, 0, N)
    ylim   = (-0.8, N-0.2)

    # Bus bars
    fig.add_trace(go.Scatter(x=[xL, xL], y=[ylim[0], ylim[1]], mode="lines", line=dict(width=3), showlegend=False))
    fig.add_trace(go.Scatter(x=[xR, xR], y=[ylim[0], ylim[1]], mode="lines", line=dict(width=3), showlegend=False))

    Imax = max(I_each) if I_each else 1.0
    for k, y in enumerate(ys):
        lw = 2 + 6*(I_each[k]/Imax)
        fig.add_trace(go.Scatter(x=[xL, x1, x2, xR], y=[y, y, y, y],
                                 mode="lines", line=dict(width=lw), showlegend=False))
    for k, y in enumerate(ys):
        fig.add_trace(go.Scatter(x=[x1], y=[y], mode="markers", marker=dict(size=10), showlegend=False))

    T = int(seconds*fps)
    speeds = [max(0.05, i/Imax) for i in I_each]
    frames = []
    for t in range(T):
        data = [go.Scatter() for _ in range(2+N)]
        for k, y in enumerate(ys):
            s = ((t/fps) * speeds[k]) % 1.0
            x = x1 + (x2-x1)*s
            data.append(go.Scatter(x=[x], y=[y]))
        frames.append(go.Frame(data=data, name=f"f{t}"))
    fig.frames = frames

    fig.update_layout(
        updatemenus=[{
            "type":"buttons",
            "buttons":[
                {"label":"▶ Play","method":"animate",
                 "args":[None,{"frame":{"duration":int(1000/fps),"redraw":True},
                               "fromcurrent":True,"transition":{"duration":0}}]},
                {"label":"⏸ Pause","method":"animate",
                 "args":[[None],{"mode":"immediate",
                                 "frame":{"duration":0,"redraw":False},
                                 "transition":{"duration":0}}]}
            ],
            "direction":"left","x":0.0,"y":1.08
        }],
        sliders=[{
            "steps":[{"args":[[f"f{t}"],{"frame":{"duration":0,"redraw":True},
                                         "mode":"immediate","transition":{"duration":0}}],
                      "label":f"{t}","method":"animate"} for t in range(T)],
            "x":0.05,"y":1.02,"len":0.9
        }],
        xaxis=dict(range=[0,10], zeroline=False, showgrid=False),
        yaxis=dict(range=list(ylim), zeroline=False, showgrid=False, scaleanchor="x", scaleratio=1),
        template="plotly_white",
        margin=dict(l=20,r=20,t=60,b=20),
        title=f"병렬 저항 애니메이션 (V={V} V, Req={Req:.3f} Ω, Itot={I_tot:.3f} A)"
    )

    annotations = []
    for k, y in enumerate(ys):
        annotations += [
            dict(x=(x1+x2)/2, y=y+0.22, text=f"R{k+1} = {Rs[k]:.3f} Ω", showarrow=False, font=dict(size=12)),
            dict(x=(x1+x2)/2, y=y-0.22, text=f"I{k+1} = {I_each[k]:.3f} A", showarrow=False, font=dict(size=12))
        ]
    fig.update_layout(annotations=annotations)
    return fig
