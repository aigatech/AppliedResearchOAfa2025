from moviepy.editor import ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips

def make_video(flyer_png, captions):
    base = ImageClip(flyer_png).set_duration(12).resize(height=720)
    zoom = base.fx(vfx.zoom_in, 1.05)  # or custom frame-by-frame scaling
    clips = []
    t=0
    for cap in captions:  # e.g., [headline, bullets, CTA]
        txt = TextClip(cap, fontsize=48, font="Inter-Bold").set_position("center").set_duration(4).set_start(t)
        clips.append(txt)
        t+=4
    comp = CompositeVideoClip([zoom, *clips])
    comp.write_videofile("promo.mp4", fps=24, audio=False)
    return "promo.mp4"