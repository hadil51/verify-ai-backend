import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from skimage import transform
from sklearn.decomposition import PCA


class RotatedBox(object):
    def __init__(self, center, width, height, angle, points=None):
        self.center = np.asarray(center, dtype=np.float64)
        self.width = width
        self.height = height
        self.angle = angle
        self.points = points

    def __repr__(self):
        return "RotatedBox(cx={0}, cy={1}, width={2}, height={3}, angle={4})".format(
            self.cx, self.cy, self.width, self.height, self.angle
        )

    @property
    def cx(self):
        return self.center[0]

    @property
    def cy(self):
        return self.center[1]

    @property
    def area(self):
        return self.width * self.height

    def rotated(self, rotation_center, angle):
        rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        t = np.asarray(rotation_center, dtype=np.float64)
        new_c = np.dot(rot.T, (self.center - t)) + t
        return RotatedBox(new_c, self.width, self.height, (self.angle + angle) % (np.pi * 2))

    def as_poly(self, margin_width=0, margin_height=0):
        v_hor = (self.width / 2 + margin_width) * np.array([np.cos(self.angle), np.sin(self.angle)])
        v_vert = (self.height / 2 + margin_height) * np.array([-np.sin(self.angle), np.cos(self.angle)])
        c = np.array([self.cx, self.cy])
        return np.vstack([c - v_hor - v_vert, c + v_hor - v_vert, c + v_hor + v_vert, c - v_hor + v_vert])

    def plot(self, mode="image", ax=None, **kwargs):
        ax = ax or plt.gca()
        poly = self.as_poly()
        if mode == "image":
            poly = poly[:, [1, 0]]
        kwargs.setdefault("fill", False)
        kwargs.setdefault("color", "r")
        kwargs.setdefault("lw", 2)
        p = patches.Polygon(poly, **kwargs)
        ax.add_patch(p)
        return p

    def extract_from_image(self, img, scale=1.0, margin_width=5, margin_height=5):
        rotate_by = (np.pi / 2 - self.angle) * 180 / np.pi
        img_rotated = transform.rotate(
            img,
            angle=rotate_by,
            center=[self.center[1] * scale, self.center[0] * scale],
            resize=True,
        )
        shift_c, shift_r = self._compensate_rotation_shift(img, scale)

        r1 = max(int((self.center[0] - self.height / 2 - margin_height) * scale - shift_r), 0)
        r2 = int((self.center[0] + self.height / 2 + margin_height) * scale - shift_r)
        c1 = max(int((self.center[1] - self.width / 2 - margin_width) * scale - shift_c), 0)
        c2 = int((self.center[1] + self.width / 2 + margin_width) * scale - shift_c)
        return img_rotated[r1:r2, c1:c2]

    def _compensate_rotation_shift(self, img, scale):
        ctr = np.asarray([self.center[1] * scale, self.center[0] * scale])
        tform1 = transform.SimilarityTransform(translation=ctr)
        tform2 = transform.SimilarityTransform(rotation=np.pi / 2 - self.angle)
        tform3 = transform.SimilarityTransform(translation=-ctr)
        tform = tform3 + tform2 + tform1

        rows, cols = img.shape[0], img.shape[1]
        corners = np.array([[0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0]])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()

        translation = (minc, minr)
        tform4 = transform.SimilarityTransform(translation=translation)
        tform = tform4 + tform
        tform.params[2] = (0, 0, 1)

        return (ctr - tform.inverse(ctr)).ravel().tolist()

    @staticmethod
    def from_points(points, box_type="bb"):
        points = np.asarray(points, dtype=np.float64)
        if points.shape[0] == 1:
            return RotatedBox(points[0], width=0.0, height=0.0, angle=0.0, points=points)

        m = PCA(2).fit(points)
        angle = np.arctan2(m.components_[0, 1], m.components_[0, 0]) % np.pi
        if abs(angle - np.pi) < angle:
            angle = angle - np.pi if angle > 0 else angle + np.pi
        points_transformed = m.transform(points)
        ll = np.min(points_transformed, 0)
        ur = np.max(points_transformed, 0)
        wh = ur - ll

        if box_type == "bb" or (box_type == "mrz" and points.shape[0] < 10):
            return RotatedBox(np.dot(m.components_.T, (ll + ur) / 2) + m.mean_, width=wh[0], height=wh[1], angle=angle, points=points)
        elif box_type == "mrz":
            h_coord = sorted(points_transformed[:, 1])
            n = len(h_coord)
            bottom, top = h_coord[n // 10], h_coord[(n * 9) // 10]
            valid_points = np.logical_and(points_transformed[:, 1] >= bottom, points_transformed[:, 1] <= top)
            rb = RotatedBox.from_points(points[valid_points, :], "bb")
            rb.points = points
            return rb
        else:
            raise ValueError("Unknown parameter value: box_type=%s" % box_type)

