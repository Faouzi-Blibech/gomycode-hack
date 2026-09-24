from s2c.partspec.schema import topology_json_schema

SYSTEM = """You describe the TOPOLOGY of a single mechanical part seen in an image.
You NEVER output a length, width, height, thickness, diameter, radius, depth or any measurement.
Positions are fractions of the part's bounding box: u from left 0 to right 1, v from bottom 0 to top 1.

The only part types are:
- plate: a flat rectangular or rounded-rectangular plate, possibly with holes or slots
- l_bracket: two flat legs joined at an angle
- flange: a round disc with a central bore and a ring of bolt holes
- spacer: a plain cylinder or tube
- profile_extrusion: any other closed flat outline extruded straight
- unsupported: anything that is not a flat profile extruded straight (curved surfaces, assemblies, threads, organic shapes)

Return ONLY a JSON object matching this schema, no prose, no code fences:
""" + topology_json_schema()

USER_BY_KIND = {
    "sketch": "This is a hand-drawn sketch on paper. Written numbers are dimensions; count them in annotation_count but do NOT report their values. Describe the topology.",
    "photo": "This is a top-down photo of a real part next to a coin. Ignore the coin. Describe the topology of the part.",
    "drawing": "This is a technical drawing. Count dimension labels in annotation_count but do NOT report their values. Describe the topology.",
}

RETRY_SUFFIX = "\n\nYour previous output failed validation:\n{error}\nReturn corrected JSON only. Remember: no measurements of any kind."
