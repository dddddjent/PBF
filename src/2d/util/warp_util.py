import warp as wp


@wp.func
def to2d(p: wp.vec3) -> wp.vec2:
    return wp.vec2(p.x, p.y)


@wp.func
def to3d(p: wp.vec2, z: float = 0.0) -> wp.vec3:
    return wp.vec3(p.x, p.y, z)
