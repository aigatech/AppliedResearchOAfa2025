import taichi as ti
import numpy as np

# --- Helper Functions (moved inside class where needed) ---

@ti.func
def hex_to_rgb_float(c: ti.i32) -> ti.math.vec3:
    return ti.math.vec3(((c >> 16) & 0xFF) / 255.0, ((c >> 8) & 0xFF) / 255.0, (c & 0xFF) / 255.0)

@ti.data_oriented
class TextureVisualizer:
    # Perlin Noise Implementation (moved inside class)
    @ti.func
    def hash_coords(self, p: ti.math.vec2) -> ti.f32:
        # A common deterministic hash for Perlin noise
        p = ti.math.vec2(ti.math.dot(p, ti.math.vec2(127.1, 311.7)),
                         ti.math.dot(p, ti.math.vec2(269.5, 183.3)))
        return ti.math.fract(ti.sin(p.x) * ti.cos(p.y) * 43758.5453123)

    @ti.func
    def grad(self, hash_val: ti.f32, x: ti.f32, y: ti.f32) -> ti.f32:
        # Gradient vectors based on hash value (standard 8 directions)
        result = 0.0
        g_x = 0.0
        g_y = 0.0
        
        h_int = ti.cast(hash_val * 8.0, ti.i32) # Map hash to 0-7
        
        if h_int == 0: g_x, g_y = 1.0, 1.0
        elif h_int == 1: g_x, g_y = -1.0, 1.0
        elif h_int == 2: g_x, g_y = 1.0, -1.0
        elif h_int == 3: g_x, g_y = -1.0, -1.0
        elif h_int == 4: g_x, g_y = 1.0, 0.0
        elif h_int == 5: g_x, g_y = -1.0, 0.0
        elif h_int == 6: g_x, g_y = 0.0, 1.0
        else: g_x, g_y = 0.0, -1.0 # h_int == 7
        
        result = g_x * x + g_y * y
        return result

    @ti.func
    def perlin_noise(self, p: ti.math.vec2) -> ti.f32:
        # Perlin noise implementation
        p0 = ti.math.floor(p)
        p1 = p0 + 1.0
        
        f = ti.math.fract(p)
        f = f * f * (3.0 - 2.0 * f) # Fade function

        # Gradients at corners
        g00 = self.grad(self.hash_coords(p0), f.x, f.y)
        g10 = self.grad(self.hash_coords(ti.math.vec2(p1.x, p0.y)), f.x - 1.0, f.y)
        g01 = self.grad(self.hash_coords(ti.math.vec2(p0.x, p1.y)), f.x, f.y - 1.0)
        g11 = self.grad(self.hash_coords(p1), f.x - 1.0, f.y - 1.0)

        # Interpolate
        return ti.math.mix(ti.math.mix(g00, g10, f.x), ti.math.mix(g01, g11, f.x), f.y)

    # Fractional Brownian Motion (FBM) using Perlin noise
    @ti.func
    def fbm(self, p: ti.math.vec2, octaves: ti.i32, frequency: ti.f32, amplitude: ti.f32) -> ti.f32:
        total = 0.0
        current_amplitude = amplitude
        current_frequency = frequency
        for i in range(octaves):
            total += self.perlin_noise(p * current_frequency) * current_amplitude
            current_frequency *= 2.0
            current_amplitude *= 0.5
        return total

    def __init__(self, grid_res=(256, 256)):
        self.grid_res = grid_res
        self.num_vertices = grid_res[0] * grid_res[1]

        # --- Grid and Mesh Data ---
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        num_triangles = (grid_res[0] - 1) * (grid_res[1] - 1) * 2
        self.indices = ti.field(ti.i32, shape=num_triangles * 3)
        self.initialize_mesh()

        # --- Sensation Parameters (Core Texture Values) ---
        self.time = ti.field(dtype=ti.f32, shape=())
        self.structure_type_id = ti.field(dtype=ti.i32, shape=())
        self.displacement_amount = ti.field(dtype=ti.f32, shape=())
        self.element_density = ti.field(dtype=ti.f32, shape=())
        self.element_scale = ti.field(dtype=ti.f32, shape=())
        self.roughness_amount = ti.field(dtype=ti.f32, shape=())
        self.glossiness = ti.field(dtype=ti.f32, shape=())
        self.metallic = ti.field(dtype=ti.f32, shape=())
        self.palette = ti.Vector.field(3, dtype=ti.f32, shape=3)

        self.structure_map = {
            "Continuous": 1, "Particulate": 2, "Fibrous": 3
        }

    @ti.kernel
    def initialize_mesh(self):
        for i, j in ti.ndrange(self.grid_res[0], self.grid_res[1]):
            idx = i * self.grid_res[1] + j
            self.vertices[idx] = ti.math.vec3((i / self.grid_res[0] - 0.5) * 5, 0, (j / self.grid_res[1] - 0.5) * 5)
        for i, j in ti.ndrange(self.grid_res[0] - 1, self.grid_res[1] - 1):
            quad_id = i * (self.grid_res[1] - 1) + j
            idx_tl, idx_tr = i * self.grid_res[1] + j, i * self.grid_res[1] + j + 1
            idx_bl, idx_br = (i + 1) * self.grid_res[1] + j, (i + 1) * self.grid_res[1] + j + 1
            self.indices[quad_id * 6 + 0], self.indices[quad_id * 6 + 1], self.indices[quad_id * 6 + 2] = idx_tl, idx_tr, idx_bl
            self.indices[quad_id * 6 + 3], self.indices[quad_id * 6 + 4], self.indices[quad_id * 6 + 5] = idx_tr, idx_br, idx_bl

    def apply_sensation(self, data: dict):
        self.structure_type_id[None] = self.structure_map.get(data.get("structure_type", "Continuous"), 1)
        self.displacement_amount[None] = data.get("displacement_amount", 0.5)
        self.element_density[None] = data.get("element_density", 0.0)
        self.element_scale[None] = data.get("element_scale", 0.5)
        self.roughness_amount[None] = data.get("roughness_amount", 0.5)
        self.glossiness[None] = data.get("glossiness", 0.5)
        self.metallic[None] = data.get("metallic", 0.0)
        colors_hex = data.get("color_palette_hex", ["#888888", "#555555", "#A0A0A0"])
        colors_int = [int(c.lstrip('#'), 16) for c in colors_hex]
        self._update_palette_kernel(colors_int[0], colors_int[1 % len(colors_int)], colors_int[2 % len(colors_int)])

    @ti.kernel
    def _update_palette_kernel(self, c1: ti.i32, c2: ti.i32, c3: ti.i32):
        self.palette[0], self.palette[1], self.palette[2] = hex_to_rgb_float(c1), hex_to_rgb_float(c2), hex_to_rgb_float(c3)

    @ti.kernel
    def _update_step(self):
        self.time[None] += 0.01
        t = self.time[None]
        
        struct_type = self.structure_type_id[None]
        disp_amt = self.displacement_amount[None]
        elem_dens = self.element_density[None]
        elem_scale = self.element_scale[None]
        rough_amt = self.roughness_amount[None]
        gloss = self.glossiness[None]
        metal = self.metallic[None]

        for i in range(self.num_vertices):
            pos = self.vertices[i]
            original_x, original_z = pos.x, pos.z
            y = 0.0
            color_mix = 0.5
            base_height = 0.0  # Track base terrain height

            # Base noise for all types
            # Note: perlin_noise and fbm are now methods of self

            if struct_type == 1: # Continuous (Concrete, Sand, Wood, Glass, etc.)
                # Dynamic surface complexity based on material properties
                # High glossiness + low roughness + low displacement = smooth transparent materials
                # Low glossiness + high roughness = rough textured materials
                
                smoothness_factor = gloss * (1.0 - rough_amt) * (1.0 - disp_amt)
                
                # Adaptive frequency and noise layers based on material properties
                base_freq = 0.1 + rough_amt * 2.0 + (1.0 - gloss) * 1.5
                noise_layers = 1 + int(rough_amt * 3) + int((1.0 - gloss) * 2)
                amplitude_scale = disp_amt * (0.1 + rough_amt * 0.9)
                
                if smoothness_factor > 0.6:  # Very smooth materials
                    # Minimal surface variation
                    base_noise = self.fbm(pos.xz * base_freq, max(1, noise_layers - 2), 0.3, amplitude_scale * 0.2)
                    micro_detail = self.perlin_noise(pos.xz * 60.0) * rough_amt * 0.01
                    y = base_noise + micro_detail
                    color_mix = 0.1 + smoothness_factor * 0.3
                    
                elif smoothness_factor > 0.3:  # Medium smooth materials
                    # Moderate surface variation
                    base_noise = self.fbm(pos.xz * base_freq, max(2, noise_layers - 1), 0.6, amplitude_scale * 0.5)
                    surface_detail = self.perlin_noise(pos.xz * (20.0 + rough_amt * 20.0)) * rough_amt * 0.05
                    y = base_noise + surface_detail
                    color_mix = 0.3 + (1.0 - rough_amt) * 0.3
                    
                else:  # Rough textured materials
                    # Full complexity with multiple noise layers
                    base_noise = self.fbm(pos.xz * base_freq, noise_layers, 1.0, amplitude_scale * 0.6)
                    medium_noise = self.fbm(pos.xz * (base_freq * 4.0), 3, 1.0, amplitude_scale * 0.3)
                    fine_noise = self.fbm(pos.xz * (10.0 + rough_amt * 40.0), 2, 1.0, rough_amt * 0.1)
                    
                    y = base_noise + medium_noise + fine_noise
                    
                    # Color mixing based on height and surface variation
                    height_factor = ti.math.clamp((y / (amplitude_scale + 1e-6)) * 0.5 + 0.5, 0.0, 1.0)
                    surface_variation = ti.math.clamp(fine_noise / (rough_amt * 0.1 + 1e-6) * 0.5 + 0.5, 0.0, 1.0)
                    color_mix = height_factor * 0.7 + surface_variation * 0.3

            elif struct_type == 2: # Particulate (Gravel, Sand grains, Dust)
                # Base terrain with gentle undulation
                base_freq = 1.0 + elem_scale * 2.0
                base_height = self.fbm(pos.xz * base_freq, 3, 1.0, disp_amt * 0.4)
                
                # Particle distribution pattern
                particle_freq = 8.0 + elem_scale * 20.0
                particle_noise = self.perlin_noise(pos.xz * particle_freq)
                
                # Secondary particle layer for more complexity
                particle_freq2 = 15.0 + elem_scale * 30.0
                particle_noise2 = self.perlin_noise(pos.xz * particle_freq2) * 0.5
                
                # Combined particle presence
                particle_presence = (particle_noise + particle_noise2) * 0.6
                
                # Density threshold for particle visibility
                density_threshold = 1.0 - elem_dens * 0.7
                
                if particle_presence > density_threshold:
                    # Particle height varies with noise intensity
                    particle_intensity = (particle_presence - density_threshold) / (1.0 - density_threshold)
                    particle_height = particle_intensity * disp_amt * 0.6
                    
                    # Add surface roughness to particles
                    surface_detail = self.perlin_noise(pos.xz * 50.0) * rough_amt * 0.05
                    
                    y = base_height + particle_height + surface_detail
                    color_mix = ti.math.clamp(particle_intensity * 0.8 + 0.2, 0.0, 1.0)
                else:
                    # Between particles - show base terrain
                    y = base_height + self.perlin_noise(pos.xz * 30.0) * rough_amt * 0.03
                    color_mix = 0.3  # Darker areas between particles

            elif struct_type == 3: # Fibrous (Grass, Fur, Carpet)
                # Highly improved grass blade generation for dense, realistic coverage
                
                # Base terrain (very subtle for grass)
                base_freq = 0.8
                base_height = self.fbm(pos.xz * base_freq, 2, 1.0, disp_amt * 0.15)
                
                # Primary grass blade clusters (creates natural patches)
                primary_freq = 12.0 + (1.0 - elem_scale) * 25.0
                primary_pattern = self.perlin_noise(pos.xz * primary_freq)
                
                # Secondary blade layer (fills in gaps for density)
                secondary_freq = 18.0 + (1.0 - elem_scale) * 35.0
                secondary_pattern = self.perlin_noise(pos.xz * secondary_freq)
                
                # Tertiary fine blade details (individual blade variation)
                tertiary_freq = 35.0 + (1.0 - elem_scale) * 60.0
                tertiary_pattern = self.perlin_noise(pos.xz * tertiary_freq)
                
                # Combine all patterns for comprehensive blade coverage
                combined_pattern = (primary_pattern * 0.5 + secondary_pattern * 0.3 + tertiary_pattern * 0.2)
                
                # Ultra-high density threshold for dense grass
                density_threshold = 1.0 - elem_dens * 0.95
                
                # Multiple blade presence checks for maximum density
                has_primary = primary_pattern > density_threshold
                has_secondary = secondary_pattern > density_threshold * 0.7
                has_tertiary = tertiary_pattern > density_threshold * 0.5
                
                if has_primary or has_secondary or has_tertiary:
                    # Calculate blade intensity from multiple layers
                    blade_intensity = 0.0
                    if has_primary:
                        blade_intensity += (primary_pattern - density_threshold) / (1.0 - density_threshold) * 0.6
                    if has_secondary:
                        blade_intensity += (secondary_pattern - density_threshold * 0.7) / (1.0 - density_threshold * 0.7) * 0.3
                    if has_tertiary:
                        blade_intensity += (tertiary_pattern - density_threshold * 0.5) / (1.0 - density_threshold * 0.5) * 0.1
                    
                    blade_intensity = ti.math.clamp(blade_intensity, 0.0, 1.0)
                    
                    # Grass blade height with natural variation
                    grass_height = base_height + blade_intensity * disp_amt * 0.8
                    
                    # Add micro-variation for individual blade realism
                    micro_variation = self.perlin_noise(pos.xz * 80.0) * rough_amt * 0.02
                    
                    y = grass_height + micro_variation
                    
                    # Color varies with blade density and height
                    color_mix = ti.math.clamp(blade_intensity * 0.9 + 0.1, 0.0, 1.0)
                else:
                    # Minimal soil showing through (very rare with high density)
                    soil_detail = self.perlin_noise(pos.xz * 50.0) * 0.05
                    y = base_height * 0.2 + soil_detail * rough_amt * 0.03
                    color_mix = 0.1  # Dark soil color
                
                # Add realistic wind movement for grass
                if has_primary or has_secondary:
                    wind_x = ti.sin(t * 1.2 + pos.x * 0.4) * 0.01 * disp_amt
                    wind_z = ti.cos(t * 1.5 + pos.z * 0.3) * 0.008 * disp_amt
                    y += wind_x + wind_z

            # Improved color mixing and material properties
            # Base color from palette
            base_color = ti.math.mix(self.palette[0], self.palette[1], color_mix)
            accent_color = self.palette[2]
            
            # Add variation with third color
            variation_noise = self.perlin_noise(pos.xz * 20.0)
            variation_factor = ti.math.clamp(variation_noise * 0.5 + 0.5, 0.0, 1.0)
            final_color = ti.math.mix(base_color, accent_color, variation_factor * 0.3)
            
            # Apply metallic effects
            if metal > 0.5:
                metallic_tint = ti.math.vec3(1.0, 0.95, 0.8)  # Slight golden tint for metals
                final_color = ti.math.mix(final_color, metallic_tint, metal * 0.4)
            
            # Apply glossiness effects (simplified lighting)
            # Higher glossiness = more contrast based on surface normal approximation
            if gloss > 0.3:
                # Approximate surface normal from height differences
                height_gradient = ti.abs(y - base_height) if base_height != 0.0 else ti.abs(y * 0.5)
                lighting_factor = 1.0 + height_gradient * gloss * 0.8
                final_color *= lighting_factor
            
            # Ensure colors stay in valid range
            final_color = ti.math.clamp(final_color, 0.0, 1.0)

            self.vertices[i].y = y
            self.colors[i] = final_color

    def update(self):
        self._update_step()

    def render(self, scene):
        scene.mesh(self.vertices, indices=self.indices, per_vertex_color=self.colors)