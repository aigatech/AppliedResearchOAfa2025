from PIL import Image, ImageDraw, ImageFont

def make_flyer(headline, tagline, bullets, cta, colors, bg_path=None):
    W, H = 1080, 1350
    img = Image.open(bg_path).resize((W,H)).convert("RGB") if bg_path else Image.new("RGB",(W,H), colors["bg"])
    draw = ImageDraw.Draw(img)
    overlay = Image.new("RGBA",(W,H),(0,0,0,90)); img.paste(overlay,(0,0),overlay)
    # load fonts from assets
    # draw headline/tagline/bullets/cta with padding and line breaks
    out_png = "flyer.png"; img.save(out_png, "PNG")
    return out_png