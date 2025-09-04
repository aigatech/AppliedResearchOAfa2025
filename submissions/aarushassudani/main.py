import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
import taichi as ti
import taichi.ui as ti_ui
from hf_handler import SensationGenerator
from visualization import TextureVisualizer # Import the new class

def main():
    """
    Main function for the SynaSight application with procedural textures.
    """
    ti.init(arch=ti.gpu)
    print("--- Welcome to SynaSight (Procedural Textures) ---")

    try:
        sensation_generator = SensationGenerator()
        visualizer = TextureVisualizer(grid_res=(256, 256))

        window = ti_ui.Window("SynaSight - Procedural Textures", res=(1280, 720), vsync=True)
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti_ui.Camera()
        camera.position(0, 3, 3)
        camera.lookat(0, 0, 0)

    except Exception as e:
        print(f"\n--- FATAL ERROR DURING INITIALIZATION ---")
        print(f"An error occurred: {e}")
        return

    # --- Initial Sensation (using default to ensure immediate GUI responsiveness) ---
    initial_data = sensation_generator._get_default_sensation()
    visualizer.apply_sensation(initial_data)

    print("\n--- Application is running ---")
    print("Describe a ground material in the console when prompted.")
    print("Press ESC to exit.")

    # --- Main Application Loop ---
    while window.running:
        if window.get_event(ti_ui.PRESS):
            if window.event.key == 'i':
                print("\n--------------------------------------------------")
                new_input = input("Describe a material (e.g., 'soft grass', 'wet sand'): ")
                print("------------------------------------------------------------------")
                if new_input:
                    print("Generating sensation data with AI model... This may take a while.")
                    sensation_data = sensation_generator.generate_data(new_input)
                    print("AI generation complete. Applying new texture.")
                    visualizer.apply_sensation(sensation_data)
            elif window.event.key == ti_ui.ESCAPE:
                window.running = False
        
        scene.set_camera(camera)

        scene.ambient_light((0.3, 0.3, 0.3))
        scene.point_light(pos=(2, 2, 2), color=(0.7, 0.7, 0.7))

        visualizer.update()
        visualizer.render(scene)

        canvas.scene(scene)
        
        window.show()

    print("Exiting SynaSight.")

if __name__ == "__main__":
    main()