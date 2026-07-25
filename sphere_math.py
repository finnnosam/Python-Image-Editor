"""
sphere_math.py
==============

Math helpers for PyPaint's globe viewer.

Part 1
------

This module intentionally contains *no* Tkinter, PIL or rendering code.
It only performs coordinate conversions and ray/sphere math.

Coordinate system
-----------------

        +Y (north)
         |
         |
-Z ------+------ +Z
         |
         |
        -Y

+X points toward longitude 0°.

The globe renderer will rotate vectors into camera space; this module
only knows about the sphere itself.

Part 2 will add:

    * great-circle interpolation
    * seam-aware stroke sampling
    * spherical brush helpers
"""

from __future__ import annotations

from dataclasses import dataclass
import math

EPSILON = 1e-9
TAU = math.pi * 2.0


# --------------------------------------------------------------------
# Vector
# --------------------------------------------------------------------

@dataclass(slots=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Vec3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def __sub__(self, other):
        return Vec3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )

    def __mul__(self, scalar: float):
        return Vec3(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar
        )

    __rmul__ = __mul__

    def dot(self, other) -> float:
        return (
            self.x * other.x +
            self.y * other.y +
            self.z * other.z
        )

    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self):
        l = self.length()

        if l < EPSILON:
            return Vec3(0.0, 0.0, 0.0)

        return Vec3(
            self.x / l,
            self.y / l,
            self.z / l
        )


# --------------------------------------------------------------------
# Clamp / wrapping helpers
# --------------------------------------------------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap01(x):
    """
    Wrap into [0,1).
    """
    return x % 1.0


def wrap_angle(theta):
    """
    Wrap radians into [-pi,+pi]
    """

    while theta <= -math.pi:
        theta += TAU

    while theta > math.pi:
        theta -= TAU

    return theta


# --------------------------------------------------------------------
# Sphere <-> latitude / longitude
# --------------------------------------------------------------------

def latlon_to_vec(latitude, longitude):
    """
    latitude:
        +pi/2 north pole
        -pi/2 south pole

    longitude:
        -pi ... +pi
    """

    clat = math.cos(latitude)

    return Vec3(
        clat * math.cos(longitude),
        math.sin(latitude),
        clat * math.sin(longitude)
    )


def vec_to_latlon(v: Vec3):
    """
    Returns

        latitude
        longitude
    """

    v = v.normalized()

    latitude = math.asin(clamp(v.y, -1.0, 1.0))

    longitude = math.atan2(
        v.z,
        v.x
    )

    return latitude, longitude


# --------------------------------------------------------------------
# Equirectangular mapping
# --------------------------------------------------------------------

def uv_to_latlon(u, v):
    """
    UV
    ---
    u = 0 left edge
    u = 1 right edge

    v = 0 north pole
    v = 1 south pole
    """

    longitude = (u * TAU) - math.pi

    latitude = (0.5 - v) * math.pi

    return latitude, longitude


def latlon_to_uv(latitude, longitude):
    """
    Returns wrapped UV coordinates.
    """

    u = (longitude + math.pi) / TAU

    v = 0.5 - (latitude / math.pi)

    return (
        wrap01(u),
        clamp(v, 0.0, 1.0)
    )


def uv_to_vec(u, v):
    lat, lon = uv_to_latlon(u, v)
    return latlon_to_vec(lat, lon)


def vec_to_uv(v):
    lat, lon = vec_to_latlon(v)
    return latlon_to_uv(lat, lon)


# --------------------------------------------------------------------
# Image coordinate conversions
# --------------------------------------------------------------------

def image_to_uv(x, y, width, height):

    return (
        x / width,
        y / height
    )


def uv_to_image(u, v, width, height):

    return (
        u * width,
        v * height
    )


def image_to_vec(x, y, width, height):

    return uv_to_vec(
        x / width,
        y / height
    )


def vec_to_image(v, width, height):

    u, vv = vec_to_uv(v)

    return (
        u * width,
        vv * height
    )


# --------------------------------------------------------------------
# Ray / sphere
# --------------------------------------------------------------------

def ray_sphere_intersection(
    origin: Vec3,
    direction: Vec3,
    radius=1.0
):
    """
    Returns the nearest hit point on the sphere.

    None if the ray misses.
    """

    direction = direction.normalized()

    a = direction.dot(direction)

    b = 2.0 * origin.dot(direction)

    c = origin.dot(origin) - radius * radius

    disc = b * b - 4 * a * c

    if disc < 0.0:
        return None

    s = math.sqrt(disc)

    t0 = (-b - s) / (2 * a)
    t1 = (-b + s) / (2 * a)

    t = None

    if t0 > 0:
        t = t0
    elif t1 > 0:
        t = t1
    else:
        return None

    return origin + direction * t


# --------------------------------------------------------------------
# Camera helpers
# --------------------------------------------------------------------

def screen_to_ndc(
    px,
    py,
    width,
    height
):
    """
    Convert screen pixel into normalized device coordinates.

    Returns

        x in [-1,+1]
        y in [-1,+1]
    """

    x = (
        (px + 0.5) / width
    ) * 2.0 - 1.0

    y = 1.0 - (
        (py + 0.5) / height
    ) * 2.0

    return x, y


def make_camera_ray(
    px,
    py,
    width,
    height,
    fov_deg=45.0
):
    """
    Camera is assumed to sit at

        (0,0,+distance)

    looking toward the origin.

    Globe window can rotate this ray later.
    """

    x, y = screen_to_ndc(
        px,
        py,
        width,
        height
    )

    aspect = width / height

    scale = math.tan(
        math.radians(fov_deg) * 0.5
    )

    dx = x * aspect * scale
    dy = y * scale
    dz = -1.0

    return Vec3(
        dx,
        dy,
        dz
    ).normalized()

# --------------------------------------------------------------------
# Rotation helpers
# --------------------------------------------------------------------

def rotate_x(v: Vec3, angle: float) -> Vec3:
    """Rotate a vector around +X."""
    c = math.cos(angle)
    s = math.sin(angle)

    return Vec3(
        v.x,
        c * v.y - s * v.z,
        s * v.y + c * v.z,
    )


def rotate_y(v: Vec3, angle: float) -> Vec3:
    """Rotate a vector around +Y."""
    c = math.cos(angle)
    s = math.sin(angle)

    return Vec3(
        c * v.x + s * v.z,
        v.y,
        -s * v.x + c * v.z,
    )


def rotate_z(v: Vec3, angle: float) -> Vec3:
    """Rotate a vector around +Z."""
    c = math.cos(angle)
    s = math.sin(angle)

    return Vec3(
        c * v.x - s * v.y,
        s * v.x + c * v.y,
        v.z,
    )


# --------------------------------------------------------------------
# Globe orientation
# --------------------------------------------------------------------

def apply_globe_rotation(v: Vec3, yaw=0.0, pitch=0.0):
    """
    Rotate the globe.

    Positive yaw spins east-west.
    Positive pitch tips the north pole upward.
    """
    return rotate_x(
        rotate_y(v, yaw),
        pitch
    )


def remove_globe_rotation(v: Vec3, yaw=0.0, pitch=0.0):
    """
    Inverse of apply_globe_rotation().
    """
    return rotate_y(
        rotate_x(v, -pitch),
        -yaw
    )


# --------------------------------------------------------------------
# Visibility
# --------------------------------------------------------------------

def is_front_facing(v: Vec3):
    """
    Assuming the camera looks down -Z and sits on +Z.

    A point is visible if it lies on the front hemisphere.
    """
    return v.z > 0.0


# --------------------------------------------------------------------
# Angular distances
# --------------------------------------------------------------------

def angular_distance(a: Vec3, b: Vec3):
    """
    Great-circle distance in radians.
    """
    d = clamp(
        a.normalized().dot(
            b.normalized()
        ),
        -1.0,
        1.0
    )

    return math.acos(d)


# --------------------------------------------------------------------
# Great-circle interpolation
# --------------------------------------------------------------------

def slerp(a: Vec3, b: Vec3, t: float):
    """
    Spherical linear interpolation.
    """

    a = a.normalized()
    b = b.normalized()

    dot = clamp(a.dot(b), -1.0, 1.0)

    if dot > 0.9999:
        return (
            a * (1.0 - t) +
            b * t
        ).normalized()

    theta = math.acos(dot)

    s = math.sin(theta)

    wa = math.sin((1.0 - t) * theta) / s
    wb = math.sin(t * theta) / s

    return (
        a * wa +
        b * wb
    ).normalized()


# --------------------------------------------------------------------
# Stroke sampling
# --------------------------------------------------------------------

def sample_arc(
    start: Vec3,
    end: Vec3,
    step_radians=math.radians(0.25)
):
    """
    Sample evenly along the shortest path on the sphere.
    """

    angle = angular_distance(start, end)

    if angle < EPSILON:
        return [start]

    # One segment is enough when the endpoints are already closer than the
    # requested brush spacing.  Callers commonly retain the previous endpoint,
    # so forcing two segments merely produces a redundant midpoint stamp.
    count = max(
        1,
        int(math.ceil(angle / step_radians))
    )

    pts = []

    for i in range(count + 1):
        t = i / count
        pts.append(
            slerp(start, end, t)
        )

    return pts


# --------------------------------------------------------------------
# Seam helpers
# --------------------------------------------------------------------

def unwrap_u(u0, u1):
    """
    Makes two U coordinates continuous.

    Example

        0.99 -> 1.01

    instead of

        0.99 -> 0.01
    """

    d = u1 - u0

    if d > 0.5:
        u1 -= 1.0

    elif d < -0.5:
        u1 += 1.0

    return u0, u1


def wrap_image_x(x, width):
    """
    Horizontal wrapping for equirectangular maps.
    """
    return x % width


# --------------------------------------------------------------------
# UV stroke conversion
# --------------------------------------------------------------------

def arc_to_uv(
    start: Vec3,
    end: Vec3,
    step_radians=math.radians(0.25)
):
    """
    Convert a spherical stroke into UV samples.
    """

    pts = sample_arc(
        start,
        end,
        step_radians
    )

    out = []

    prev_u = None

    for p in pts:

        u, v = vec_to_uv(p)

        if prev_u is not None:
            _, u = unwrap_u(prev_u, u)

        out.append((u, v))
        prev_u = u

    return out


# --------------------------------------------------------------------
# Brush footprint
# --------------------------------------------------------------------

def spherical_brush_points(
    center: Vec3,
    angular_radius,
    rings=5,
    segments=32
):
    """
    Generate sample points inside a circular brush on
    the sphere.

    Returns Vec3 samples.
    """

    center = center.normalized()

    north = Vec3(0, 1, 0)

    if abs(center.dot(north)) > 0.95:
        north = Vec3(1, 0, 0)

    tangent = north.cross(center).normalized()
    bitangent = center.cross(tangent).normalized()

    pts = [center]

    for r in range(1, rings + 1):

        rr = angular_radius * (r / rings)

        sinr = math.sin(rr)
        cosr = math.cos(rr)

        for i in range(segments):

            a = TAU * i / segments

            direction = (
                tangent * math.cos(a) +
                bitangent * math.sin(a)
            )

            p = (
                center * cosr +
                direction * sinr
            ).normalized()

            pts.append(p)

    return pts


# --------------------------------------------------------------------
# Brush conversion
# --------------------------------------------------------------------

def spherical_brush_uv(
    center: Vec3,
    angular_radius,
    rings=5,
    segments=32
):
    """
    Convenience wrapper.

    Returns UV coordinates for every sample point.
    """

    return [
        vec_to_uv(v)
        for v in spherical_brush_points(
            center,
            angular_radius,
            rings,
            segments
        )
    ]


# --------------------------------------------------------------------
# Picking
# --------------------------------------------------------------------

def pick_uv(
    ray_origin: Vec3,
    ray_direction: Vec3
):
    """
    Cast a ray at the globe.

    Returns

        (u,v)

    or

        None
    """

    hit = ray_sphere_intersection(
        ray_origin,
        ray_direction
    )

    if hit is None:
        return None

    return vec_to_uv(hit)
