"""Blender-only worker for pendant.py; all scene assets are generated locally."""

import json
import math
from pathlib import Path
import sys

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def material(name, color, metallic, roughness):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return result, shader


def gold_material():
    gold, shader = material("Polished warm yellow gold", (0.83, 0.58, 0.24), 1, 0.2)
    nodes, links = gold.node_tree.nodes, gold.node_tree.links
    coordinates = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 70000
    noise.inputs["Detail"].default_value = 2
    links.new(coordinates.outputs["Object"], noise.inputs["Vector"])
    roughness = nodes.new("ShaderNodeMapRange")
    roughness.inputs["To Min"].default_value = 0.17
    roughness.inputs["To Max"].default_value = 0.24
    links.new(noise.outputs["Fac"], roughness.inputs["Value"])
    links.new(roughness.outputs["Result"], shader.inputs["Roughness"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.000002
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return gold


def curve_object(name, radius, gold):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 3
    curve.bevel_depth = radius
    curve.bevel_resolution = 5
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    curve.materials.append(gold)
    return obj


def word_curve(word, radius, gold):
    obj = curve_object(word["name"], radius, gold)
    spline = obj.data.splines.new("BEZIER")
    cubics = word["cubics"]
    spline.bezier_points.add(len(cubics) - 1)
    for i, (point, segment) in enumerate(zip(spline.bezier_points, cubics)):
        point.handle_left_type = point.handle_right_type = "FREE"
        point.co = segment[0]
        point.handle_left = cubics[i - 1][2]
        point.handle_right = segment[1]
    spline.use_cyclic_u = True
    return obj


def oval(name, center, axis_a, axis_b, radius, gold):
    obj = curve_object(name, radius, gold)
    obj.data.resolution_u = 24
    spline = obj.data.splines.new("BEZIER")
    spline.bezier_points.add(3)
    center, a, b = Vector(center), Vector(axis_a), Vector(axis_b)
    for i, point in enumerate(spline.bezier_points):
        t = i * math.pi / 2
        position = center + a * math.cos(t) + b * math.sin(t)
        tangent = (-a * math.sin(t) + b * math.cos(t)) * (4 * (math.sqrt(2) - 1) / 3)
        point.handle_left_type = point.handle_right_type = "FREE"
        point.co = position
        point.handle_left = position - tangent
        point.handle_right = position + tangent
    spline.use_cyclic_u = True
    return obj


def mesh_components(mesh):
    neighbors = [[] for _ in mesh.vertices]
    for edge in mesh.edges:
        a, b = edge.vertices
        neighbors[a].append(b)
        neighbors[b].append(a)
    unseen = set(range(len(neighbors)))
    components = []
    while unseen:
        stack = [unseen.pop()]
        vertices = list(stack)
        while stack:
            for neighbor in neighbors[stack.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    vertices.append(neighbor)
        components.append(vertices)
    return components


def clean_voxel_shells(body, voxel_size):
    components = mesh_components(body.data)
    artifacts = []
    for vertices in components:
        if len(vertices) > 32:
            continue
        points = [body.data.vertices[index].co for index in vertices]
        extent = max(max(p[axis] for p in points) - min(p[axis] for p in points) for axis in range(3))
        if extent <= 2 * voxel_size:
            artifacts.append(vertices)
    if artifacts:
        # A real wire is eighteen voxels thick; these tiny isolated shells are remesh artifacts.
        mesh = bmesh.new()
        mesh.from_mesh(body.data)
        mesh.verts.ensure_lookup_table()
        bmesh.ops.delete(mesh, geom=[mesh.verts[index] for group in artifacts for index in group], context="VERTS")
        mesh.to_mesh(body.data)
        mesh.free()
        print(f"Removed {len(artifacts)} isolated sub-wire voxel shells ({sum(map(len, artifacts))} vertices).")
    body["removed_voxel_shells"] = len(artifacts)


def weld(objects, wire):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.convert(target="MESH")
    bpy.ops.object.join()
    body = bpy.context.object
    body.name = "One fused openwork pendant"
    remesh = body.modifiers.new("Fused crossings and solder fillets", "REMESH")
    remesh.mode = "VOXEL"
    remesh.voxel_size = wire / 18
    remesh.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier=remesh.name)
    clean_voxel_shells(body, wire / 18)
    smooth = body.modifiers.new("Polished junctions", "SMOOTH")
    smooth.factor = 0.3
    smooth.iterations = 3
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    for polygon in body.data.polygons:
        polygon.use_smooth = True
    components = mesh_components(body.data)
    if len(components) != 1:
        raise ValueError(f"Gold body has disconnected parts ({list(map(len, components))} vertices); use an intersecting composition.")
    body.location.z = 0.000015 - min(vertex.co.z for vertex in body.data.vertices)
    return body


def validate_centerlines(body, words, wire):
    tree = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())
    checked = 0
    for word in words:
        for segment in word["cubics"]:
            a, b, c, d = map(Vector, segment)
            for point in (a, (a + 3 * b + 3 * c + d) / 8):
                closest, normal, _, _ = tree.find_nearest(point)
                if closest is None or normal.dot(point - closest) > wire / 18:
                    raise ValueError(f"Fused geometry lost part of the {word['name']} centerline.")
                checked += 1
    return checked


def point_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def softbox(name, location, target, power, size, size_y, color=(1, 1, 1)):
    light = bpy.data.lights.new(name, "AREA")
    light.energy = power
    light.shape = "RECTANGLE"
    light.size, light.size_y, light.color = size, size_y, color
    obj = bpy.data.objects.new(name, light)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    point_at(obj, target)


def studio(job, gold):
    factor = job["diameter_mm"] / 32
    anchor = job["anchor_y"]
    radius = job["wire_mm"] / 2000
    body = [word_curve(word, radius, gold) for word in job["words"]]
    body.append(oval("Cast attachment eye", (0, anchor + 0.0009 * factor, radius),
                     (0.0012 * factor, 0, 0), (0, 0.0012 * factor, 0),
                     0.00026 * factor, gold))
    body = weld(body, radius * 2)
    oval("Connector ring", (0, anchor + 0.002 * factor, 0.001 * factor),
         (0, 0.0009 * factor, 0), (0, 0, 0.0008 * factor), 0.00018 * factor, gold)
    for i in range(-23, 24):
        x = i * 0.00165 * factor
        y = anchor + factor * (0.0027 + 0.017 * (abs(x) / (0.038 * factor)) ** 1.5)
        slope = 0 if i == 0 else math.copysign(0.017 * 1.5 / 0.038 * (abs(x) / (0.038 * factor)) ** 0.5, i)
        tangent = Vector((1, slope, 0)).normalized()
        across = Vector((-tangent.y, tangent.x, 0))
        tilt = math.radians(50 if i % 2 else -50)
        short = across * math.cos(tilt) + Vector((0, 0, math.sin(tilt)))
        z = factor * (0.00062 * abs(math.sin(tilt)) + 0.00018)
        oval(f"Interlocking cable link {i + 24:02d}", (x, y, z),
             tangent * (0.0012 * factor), short * (0.00062 * factor), 0.00018 * factor, gold)

    surface, shader = material("Charcoal jewelry display", (0.022, 0.027, 0.032), 0, 0.82)
    nodes, links = surface.node_tree.nodes, surface.node_tree.links
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1800
    noise.inputs["Detail"].default_value = 2
    coordinate = nodes.new("ShaderNodeTexCoord")
    links.new(coordinate.outputs["Object"], noise.inputs["Vector"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.00004
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    bpy.ops.mesh.primitive_plane_add(size=2 * factor)
    bpy.context.object.name = "Fine-grained matte display surface"
    bpy.context.object.data.materials.append(surface)

    def scaled(v):
        return tuple(value * factor for value in v)

    softbox("Large neutral key", scaled((-0.045, -0.025, 0.075)), (0, 0, 0),
            0.1 * factor**2, 0.045 * factor, 0.065 * factor)
    softbox("Narrow edge reflection", scaled((0.045, 0.012, 0.045)), (0, 0, 0),
            0.035 * factor**2, 0.008 * factor, 0.055 * factor, (0.88, 0.94, 1))
    softbox("Upper soft strip", scaled((-0.01, 0.055, 0.045)), (0, 0, 0),
            0.055 * factor**2, 0.055 * factor, 0.012 * factor)
    world = bpy.data.worlds.new("Neutral studio ambient")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.45, 0.5, 0.6, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.18
    bpy.context.scene.world = world
    bpy.ops.object.camera_add(location=scaled((0.006, -0.039, 0.135)))
    camera = bpy.context.object
    camera.name = "Focus-stacked jewelry camera"
    camera.data.lens = 85
    camera.data.clip_start = 0.001 * factor
    camera.data.clip_end = 10 * factor
    point_at(camera, scaled((0, 0.005, 0)))
    bpy.context.scene.camera = camera
    return body


def main():
    if bpy.app.version < (4, 5, 0):
        raise ValueError("Use Blender 4.5 or newer.")
    source = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    job = json.loads(source.read_text(encoding="utf-8"))
    if job["version"] != 1:
        raise ValueError("Unsupported pendant scene version.")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = job["samples"]
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.012
    scene.cycles.use_denoising = True
    scene.cycles.seed = 37
    scene.cycles.max_bounces = 12
    scene.cycles.glossy_bounces = 8
    scene.render.threads_mode = "FIXED"
    scene.render.threads = job["threads"]
    scene.render.resolution_x = scene.render.resolution_y = job["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.filepath = str(source.parent / "pendant.png")
    body = studio(job, gold_material())
    checked = validate_centerlines(body, job["words"], job["wire_mm"] / 1000)
    if job["save_scene"]:
        scene.render.filepath = "//pendant.png"
        bpy.ops.wm.save_as_mainfile(filepath=str(source.parent / "pendant.blend"), compress=True)
    bpy.ops.render.render(write_still=True)
    report = {"blender_version": bpy.app.version_string, "engine": "Cycles CPU", "seed": 37,
              "word_count": len(job["words"]), "body_components": len(mesh_components(body.data)),
              "removed_voxel_shells": body["removed_voxel_shells"],
              "source_centerline_checks": checked,
              "body_dimensions_mm": [value * 1000 for value in body.dimensions],
              "chain_links": 47, "resolution": job["resolution"], "samples": job["samples"],
              "gold": "Artistic warm yellow gold, metallic BSDF; not a certified alloy",
              "geometry": "Source cubic centerlines; round wires, voxel-fused crossings and attachment eye",
              "notice": "Rendered jewelry concept, not a photograph or manufacturing-ready model."}
    (source.parent / "render.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
