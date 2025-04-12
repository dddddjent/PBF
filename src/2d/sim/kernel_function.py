import warp as wp


# 2d
@wp.func
def W(r: float, kernel_radius: float = 1.0) -> float:
    sigma = 40.0 / (7.0 * wp.PI * kernel_radius**2.0)

    q = r / kernel_radius

    if q >= 0.0 and q <= 0.5:
        return sigma * (6.0 * (q**3.0 - q * q) + 1.0)
    elif q > 0.5 and q <= 1.0:
        return sigma * (2.0 * (1.0 - q) ** 3.0)

    return 0.0

@wp.func
def W(r: wp.vec2, kernel_radius: float = 1.0) -> float:
    sigma = 40.0 / (7.0 * wp.PI * kernel_radius**2.0)

    q = wp.length(r) / kernel_radius

    if q >= 0.0 and q <= 0.5:
        return sigma * (6.0 * (q**3.0 - q * q) + 1.0)
    elif q > 0.5 and q <= 1.0:
        return sigma * (2.0 * (1.0 - q) ** 3.0)

    return 0.0


# @wp.func
# def gradW(r: float, kernel_radius: float = 1.0) -> float:
#     sigma = 40.0 / (7.0 * wp.PI * kernel_radius**2.0)
#
#     q = wp.length(r) / kernel_radius
#
#     if q >= 0.0 and q <= 0.5:
#         return sigma * (6.0 * (3.0 * q * q - 2.0 * q))
#     elif q > 0.5 and q <= 1.0:
#         return sigma * (-6.0 * (1.0 - q) * (1.0 - q))
#
#     return 0.0
#
#
# @wp.func
# def grad2W(r: float, kernel_radius: float = 1.0) -> float:
#     sigma = 40.0 / (7.0 * wp.PI * kernel_radius**2.0)
#
#     q = wp.length(r) / kernel_radius
#
#     if q >= 0.0 and q <= 0.5:
#         return sigma * (6.0 * (6.0 * q - 2.0))
#     elif q > 0.5 and q <= 1.0:
#         return sigma * (12.0 * (1.0 - q))
#
#     return 0.0
