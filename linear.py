"""
Matrix math for OpenXR/OpenGL
Based on pyopenxr examples
"""

from ctypes import addressof, c_float, Structure
import enum
import math
import numpy

import xr


class GraphicsAPI(enum.Enum):
    VULKAN = 0,
    OPENGL = 1,
    OPENGL_ES = 2,
    D3D = 3,


class Matrix4x4f(Structure):
    """Column-major, pre-multiplied matrix"""
    _fields_ = [("m", c_float * 16), ]

    def __init__(self):
        super().__init__()
        self._numpy = None

    def __matmul__(self, other) -> "Matrix4x4f":
        return self.multiply(other)

    def as_numpy(self):
        if not hasattr(self, "_numpy") or self._numpy is None:
            buffer = (c_float * 16).from_address(addressof(self))
            buffer._wrapper = self
            self._numpy = numpy.ctypeslib.as_array(buffer)
        return self._numpy

    @staticmethod
    def create_from_quaternion(quat: xr.Quaternionf) -> "Matrix4x4f":
        x2 = quat.x + quat.x
        y2 = quat.y + quat.y
        z2 = quat.z + quat.z

        xx2 = quat.x * x2
        yy2 = quat.y * y2
        zz2 = quat.z * z2

        yz2 = quat.y * z2
        wx2 = quat.w * x2
        xy2 = quat.x * y2
        wz2 = quat.w * z2
        xz2 = quat.x * z2
        wy2 = quat.w * y2

        result = Matrix4x4f()
        result.m[0] = 1.0 - yy2 - zz2
        result.m[1] = xy2 + wz2
        result.m[2] = xz2 - wy2
        result.m[3] = 0.0
        result.m[4] = xy2 - wz2
        result.m[5] = 1.0 - xx2 - zz2
        result.m[6] = yz2 + wx2
        result.m[7] = 0.0
        result.m[8] = xz2 + wy2
        result.m[9] = yz2 - wx2
        result.m[10] = 1.0 - xx2 - yy2
        result.m[11] = 0.0
        result.m[12] = 0.0
        result.m[13] = 0.0
        result.m[14] = 0.0
        result.m[15] = 1.0
        return result

    @staticmethod
    def create_projection_fov(fov: xr.Fovf, near_z: float, far_z: float) -> "Matrix4x4f":
        tan_left = math.tan(fov.angle_left)
        tan_right = math.tan(fov.angle_right)
        tan_down = math.tan(fov.angle_down)
        tan_up = math.tan(fov.angle_up)
        
        tan_angle_width = tan_right - tan_left
        tan_angle_height = tan_up - tan_down  # OpenGL: positive Y up
        offset_z = near_z  # OpenGL: [-1,1] Z clip space
        
        result = Matrix4x4f()
        result.m[0] = 2.0 / tan_angle_width
        result.m[4] = 0.0
        result.m[8] = (tan_right + tan_left) / tan_angle_width
        result.m[12] = 0.0

        result.m[1] = 0.0
        result.m[5] = 2.0 / tan_angle_height
        result.m[9] = (tan_up + tan_down) / tan_angle_height
        result.m[13] = 0.0

        result.m[2] = 0.0
        result.m[6] = 0.0
        result.m[10] = -(far_z + offset_z) / (far_z - near_z)
        result.m[14] = -(far_z * (near_z + offset_z)) / (far_z - near_z)

        result.m[3] = 0.0
        result.m[7] = 0.0
        result.m[11] = -1.0
        result.m[15] = 0.0
        return result

    @staticmethod
    def create_scale(x: float, y: float, z: float) -> "Matrix4x4f":
        result = Matrix4x4f()
        result.m[0] = x
        result.m[5] = y
        result.m[10] = z
        result.m[15] = 1.0
        return result

    @staticmethod
    def create_translation(x: float, y: float, z: float) -> "Matrix4x4f":
        result = Matrix4x4f()
        result.m[0] = 1.0
        result.m[5] = 1.0
        result.m[10] = 1.0
        result.m[12] = x
        result.m[13] = y
        result.m[14] = z
        result.m[15] = 1.0
        return result

    @staticmethod
    def create_translation_rotation_scale(translation: xr.Vector3f, rotation: xr.Quaternionf, scale: xr.Vector3f) -> "Matrix4x4f":
        scale_matrix = Matrix4x4f.create_scale(scale.x, scale.y, scale.z)
        rotation_matrix = Matrix4x4f.create_from_quaternion(rotation)
        translation_matrix = Matrix4x4f.create_translation(translation.x, translation.y, translation.z)
        combined_matrix = rotation_matrix @ scale_matrix
        return translation_matrix @ combined_matrix

    def multiply(self, b: "Matrix4x4f") -> "Matrix4x4f":
        result = Matrix4x4f()
        result.m[0] = self.m[0] * b.m[0] + self.m[4] * b.m[1] + self.m[8] * b.m[2] + self.m[12] * b.m[3]
        result.m[1] = self.m[1] * b.m[0] + self.m[5] * b.m[1] + self.m[9] * b.m[2] + self.m[13] * b.m[3]
        result.m[2] = self.m[2] * b.m[0] + self.m[6] * b.m[1] + self.m[10] * b.m[2] + self.m[14] * b.m[3]
        result.m[3] = self.m[3] * b.m[0] + self.m[7] * b.m[1] + self.m[11] * b.m[2] + self.m[15] * b.m[3]

        result.m[4] = self.m[0] * b.m[4] + self.m[4] * b.m[5] + self.m[8] * b.m[6] + self.m[12] * b.m[7]
        result.m[5] = self.m[1] * b.m[4] + self.m[5] * b.m[5] + self.m[9] * b.m[6] + self.m[13] * b.m[7]
        result.m[6] = self.m[2] * b.m[4] + self.m[6] * b.m[5] + self.m[10] * b.m[6] + self.m[14] * b.m[7]
        result.m[7] = self.m[3] * b.m[4] + self.m[7] * b.m[5] + self.m[11] * b.m[6] + self.m[15] * b.m[7]

        result.m[8] = self.m[0] * b.m[8] + self.m[4] * b.m[9] + self.m[8] * b.m[10] + self.m[12] * b.m[11]
        result.m[9] = self.m[1] * b.m[8] + self.m[5] * b.m[9] + self.m[9] * b.m[10] + self.m[13] * b.m[11]
        result.m[10] = self.m[2] * b.m[8] + self.m[6] * b.m[9] + self.m[10] * b.m[10] + self.m[14] * b.m[11]
        result.m[11] = self.m[3] * b.m[8] + self.m[7] * b.m[9] + self.m[11] * b.m[10] + self.m[15] * b.m[11]

        result.m[12] = self.m[0] * b.m[12] + self.m[4] * b.m[13] + self.m[8] * b.m[14] + self.m[12] * b.m[15]
        result.m[13] = self.m[1] * b.m[12] + self.m[5] * b.m[13] + self.m[9] * b.m[14] + self.m[13] * b.m[15]
        result.m[14] = self.m[2] * b.m[12] + self.m[6] * b.m[13] + self.m[10] * b.m[14] + self.m[14] * b.m[15]
        result.m[15] = self.m[3] * b.m[12] + self.m[7] * b.m[13] + self.m[11] * b.m[14] + self.m[15] * b.m[15]
        return result

    def invert_rigid_body(self) -> "Matrix4x4f":
        """Calculates the inverse of a rigid body transform."""
        result = Matrix4x4f()
        result.m[0] = self.m[0]
        result.m[1] = self.m[4]
        result.m[2] = self.m[8]
        result.m[3] = 0.0
        result.m[4] = self.m[1]
        result.m[5] = self.m[5]
        result.m[6] = self.m[9]
        result.m[7] = 0.0
        result.m[8] = self.m[2]
        result.m[9] = self.m[6]
        result.m[10] = self.m[10]
        result.m[11] = 0.0
        result.m[12] = -(self.m[0] * self.m[12] + self.m[1] * self.m[13] + self.m[2] * self.m[14])
        result.m[13] = -(self.m[4] * self.m[12] + self.m[5] * self.m[13] + self.m[6] * self.m[14])
        result.m[14] = -(self.m[8] * self.m[12] + self.m[9] * self.m[13] + self.m[10] * self.m[14])
        result.m[15] = 1.0
        return result

