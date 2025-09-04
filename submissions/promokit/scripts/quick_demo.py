#!/usr/bin/env python3
"""
PromoKit Lite CLI - Command line interface for generating marketing content.
"""

import argparse
import os
import sys
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules
from core import branding, copywriter, ranker, flyer, video, images_remote, safety

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="PromoKit Lite - Generate marketing flyers and videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/quick_demo.py --prompt "Modern yoga studio in Atlanta" --tone friendly
  python scripts/quick_demo.py --url "https://business.site" --tone professional --remote-image
        """
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt", 
        type=str, 
        help="Business description prompt"
    )
    input_group.add_argument(
        "--url", 
        type=str, 
        help="Website URL to scrape"
    )
    
    # Options
    parser.add_argument(
        "--tone", 
        type=str, 
        choices=["friendly", "professional", "playful", "bold"],
        default="friendly",
        help="Brand tone (default: friendly)"
    )
    parser.add_argument(
        "--remote-image", 
        action="store_true",
        help="Use AI-generated background image (requires HF_TOKEN)"
    )
    parser.add_argument(
        "--tts", 
        action="store_true",
        help="Add text-to-speech voiceover (not implemented in MVP)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=".",
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        print("🎨 PromoKit Lite - Generating marketing content...")
        
        # 1) Get brief and palette
        brief = ""
        palette = None
        
        if args.url:
            print(f"🌐 Scraping website: {args.url}")
            
            # Validate URL
            if not safety.validate_url(args.url):
                print("❌ Invalid URL provided")
                return 1
            
            # Scrape website
            scraped_data = branding.scrape(args.url)
            if scraped_data["text"]:
                summary = branding.summarize(scraped_data["text"])
                brief = summary["summary"]
                palette = branding.extract_palette(scraped_data["text"], scraped_data["candidate_hex_colors"])
                if args.verbose:
                    print(f"📝 Extracted summary: {brief[:100]}...")
                    print(f"🎨 Found colors: {palette}")
            else:
                print("⚠️  Could not extract content from URL")
                return 1
        else:
            brief = args.prompt
        
        # Sanitize brief
        brief = safety.sanitize_text(brief)
        if not safety.is_safe(brief):
            print("❌ Content safety check failed")
            return 1
        
        # 2) Generate copy
        print("🤖 Generating copy...")
        copy_result = copywriter.gen_copy(brief, args.tone, palette_hint=palette)
        
        # Safety check
        if not safety.check_content_safety(copy_result):
            print("⚠️  Using safe fallback content")
            copy_result = safety.get_safe_fallback_content()
        
        # 3) Rank elements
        copy_result = ranker.rank_copy_elements(copy_result, goal="increase engagement")
        
        # Extract elements
        headline = copy_result["headline"]
        tagline = copy_result["tagline"]
        bullets = copy_result["bullets"]
        cta = copy_result["ctas"][0] if copy_result["ctas"] else "Get Started"
        final_palette = copy_result.get("palette", palette) or ["#2563EB", "#64748B", "#F59E0B"]
        
        if args.verbose:
            print(f"📰 Headline: {headline}")
            print(f"🏷️  Tagline: {tagline}")
            print(f"📋 Bullets: {bullets}")
            print(f"🎯 CTA: {cta}")
            print(f"🎨 Palette: {final_palette}")
        
        # 4) Generate background image (optional)
        bg_path = None
        if args.remote_image:
            print("🎨 Generating AI background image...")
            bg_prompt = f"Minimal, abstract, {args.tone} background related to: {brief[:50]}"
            bg_path = images_remote.generate_safe_background(bg_prompt)
            if bg_path:
                print("✅ Background image generated")
            else:
                print("⚠️  Background image generation failed, using gradient")
        
        # 5) Generate flyer
        print("📄 Creating flyer...")
        flyer_png_path = os.path.join(args.output_dir, "flyer.png")
        flyer_png = flyer.render_png(headline, tagline, bullets, cta, final_palette, bg_path, flyer_png_path)
        
        if flyer_png and os.path.exists(flyer_png):
            print(f"✅ Flyer PNG created: {flyer_png}")
        else:
            print("❌ Flyer PNG creation failed")
            return 1
        
        # Generate PDF
        flyer_pdf_path = os.path.join(args.output_dir, "flyer.pdf")
        flyer_pdf = flyer.render_pdf(flyer_png, flyer_pdf_path)
        
        if flyer_pdf and os.path.exists(flyer_pdf):
            print(f"✅ Flyer PDF created: {flyer_pdf}")
        else:
            print("⚠️  PDF creation failed, PNG available")
        
        # 6) Generate video
        print("🎥 Creating promo video...")
        captions = [headline] + bullets + [cta]
        video_path = os.path.join(args.output_dir, "promo.mp4")
        video_result = video.compose_slideshow(
            flyer_png, 
            captions, 
            duration_per=4, 
            tts_text=brief if args.tts else None,
            out=video_path
        )
        
        if video_result and os.path.exists(video_result):
            print(f"✅ Promo video created: {video_result}")
        else:
            print("❌ Video creation failed")
            return 1
        
        print("\n🎉 Generation completed successfully!")
        print(f"📁 Output files in: {os.path.abspath(args.output_dir)}")
        print("   - flyer.png (Flyer image)")
        print("   - flyer.pdf (Flyer PDF)")
        print("   - promo.mp4 (Promo video)")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
