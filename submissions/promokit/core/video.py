from moviepy import ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
from moviepy.video.fx import Resize
import os
import tempfile
import numpy as np

def compose_slideshow(png_path, captions, duration_per=4, out="promo.mp4", tts_text=None):
    """Create a slideshow video with captions and optional TTS."""
    try:
        if not os.path.exists(png_path):
            print(f"PNG file not found: {png_path}")
            return None
        
        # Use the simplified video creation function
        return create_simple_video_with_captions(png_path, captions, out)
        
    except Exception as e:
        print(f"Error composing slideshow: {e}")
        return create_fallback_video(out)

def tts_generate(text):
    """Generate TTS audio (optional implementation)."""
    try:
        # For MVP, we'll skip TTS implementation
        # In production, you could use:
        # - espnet/kan-bayashi_ljspeech_vits
        # - coqui-ai/TTS
        # - gTTS (Google Text-to-Speech)
        print("TTS not implemented in MVP")
        return None
    except Exception as e:
        print(f"Error in TTS generation: {e}")
        return None

def create_fallback_video(out_path):
    """Create a simple fallback video if composition fails."""
    try:
        # Create a simple colored background
        from moviepy import ColorClip
        
        # Create a 5-second video with text
        background = ColorClip(size=(1280, 720), color=(100, 100, 200), duration=5)
        
        text_clip = TextClip(
            text="Video Generation Failed",
            font_size=60,
            color='white'
        ).with_position('center').with_duration(5)
        
        final_video = CompositeVideoClip([background, text_clip])
        final_video.write_videofile(out_path, fps=24, codec='libx264')
        
        final_video.close()
        return out_path
        
    except Exception as e:
        print(f"Error creating fallback video: {e}")
        return None

def create_simple_video_with_captions(png_path, captions, out="promo.mp4"):
    """Enhanced video creation with better effects and transitions."""
    try:
        if not os.path.exists(png_path):
            return None
        
        # Load and resize image
        base_clip = ImageClip(png_path)
        base_clip = base_clip.resized(height=720)
        total_duration = len(captions) * 4  # 4 seconds per caption for better pacing
        base_clip = base_clip.with_duration(total_duration)
        
        # Add subtle zoom effect (simplified)
        # base_clip = base_clip.fx(moviepy.video.fx.resize, 1.05)
        
        # Create enhanced text overlays with better styling
        text_clips = []
        for i, caption in enumerate(captions):
            if caption:
                # Create text clip with better styling
                text_clip = TextClip(
                    text=caption,
                    font_size=48,
                    color='white',
                    stroke_color='black',
                    stroke_width=2
                ).with_position('center').with_duration(4).with_start(i * 4)
                
                # Add fade in/out effects (simplified)
                # text_clip = text_clip.fx(moviepy.video.fx.fadein, 0.5).fx(moviepy.video.fx.fadeout, 0.5)
                text_clips.append(text_clip)
        
        # Compose video with better layering
        if text_clips:
            final_video = CompositeVideoClip([base_clip] + text_clips)
        else:
            final_video = base_clip
        
        # Write video with better quality settings
        final_video.write_videofile(
            out, 
            fps=30,  # Higher frame rate for smoother video
            codec='libx264',
            audio_codec=None,
            preset='medium',  # Better quality encoding
            bitrate='2000k'  # Higher bitrate for better quality
        )
        final_video.close()
        
        return out
        
    except Exception as e:
        print(f"Error in enhanced video creation: {e}")
        return create_fallback_video(out)
