import io

import numpy as np
from skimage import filters, measure, morphology, transform
from skimage import io as skimage_io

from ..util.geometry import RotatedBox
from ..util.ocr import ocr
from ..util.pipeline import Pipeline
from .text import MRZ


class Loader(object):
    __depends__ = []
    __provides__ = ["img"]

    def __init__(self, file, as_gray=True, pdf_aware=False):
        self.file = file
        self.as_gray = as_gray
        self.pdf_aware = pdf_aware

    def _imread(self, file):
        img = skimage_io.imread(file, as_gray=self.as_gray, plugin="imageio")
        if img is not None and len(img.shape) != 2:
            img = skimage_io.imread(file, as_gray=self.as_gray, plugin="matplotlib")
        return img

    def __call__(self):
        if isinstance(self.file, str):
            return self._imread(self.file)
        elif isinstance(self.file, (bytes, io.IOBase)):
            return self._imread(self.file)
        return None


class Scaler(object):
    __depends__ = ["img"]
    __provides__ = ["img_small", "scale_factor"]

    def __init__(self, max_width=250):
        self.max_width = max_width

    def __call__(self, img):
        scale_factor = self.max_width / float(img.shape[1])
        if scale_factor <= 1:
            img_small = transform.rescale(img, scale_factor, mode="constant", channel_axis=None, anti_aliasing=True)
        else:
            scale_factor = 1.0
            img_small = img
        return img_small, scale_factor


class BooneTransform(object):
    __depends__ = ["img_small"]
    __provides__ = ["img_binary"]

    def __init__(self, square_size=5):
        self.square_size = square_size

    def __call__(self, img_small):
        m = morphology.square(self.square_size)
        img_th = morphology.black_tophat(img_small, m)
        img_sob = abs(filters.sobel_v(img_th))
        img_closed = morphology.closing(img_sob, m)
        threshold = filters.threshold_otsu(img_closed)
        return img_closed > threshold


class MRZBoxLocator(object):
    __depends__ = ["img_binary"]
    __provides__ = ["boxes"]

    def __init__(
        self,
        max_boxes=4,
        min_points_in_contour=50,
        min_area=500,
        min_box_aspect=5,
        angle_tol=0.1,
        lineskip_tol=1.5,
        box_type="bb",
    ):
        self.max_boxes = max_boxes
        self.min_points_in_contour = min_points_in_contour
        self.min_area = min_area
        self.min_box_aspect = min_box_aspect
        self.angle_tol = angle_tol
        self.lineskip_tol = lineskip_tol
        self.box_type = box_type

    def __call__(self, img_binary):
        cs = measure.find_contours(img_binary, 0.5)
        results = []
        for c in cs:
            ll, ur = np.min(c, 0), np.max(c, 0)
            wh = ur - ll
            if wh[0] * wh[1] < self.min_area:
                continue
            rb = RotatedBox.from_points(c, self.box_type)
            if rb.height == 0 or rb.width / rb.height < self.min_box_aspect:
                continue
            results.append(rb)
        results.sort(key=lambda x: -x.area)
        return self._fixup_boxes(self._merge_boxes(results[0 : self.max_boxes]))

    def _are_aligned_angles(self, b1, b2):
        return abs(b1 - b2) <= self.angle_tol or abs(np.pi - abs(b1 - b2)) <= self.angle_tol

    def _are_nearby_parallel_boxes(self, b1, b2):
        if not self._are_aligned_angles(b1.angle, b2.angle):
            return False
        angle = min(b1.angle, b2.angle)
        return abs(np.dot(b1.center - b2.center, [-np.sin(angle), np.cos(angle)])) < self.lineskip_tol * (
            b1.height + b2.height
        ) and (b1.width > 0) and (b2.width > 0) and (0.5 < b1.width / b2.width < 2.0)

    def _merge_any_two_boxes(self, box_list):
        n = len(box_list)
        for i in range(n):
            for j in range(i + 1, n):
                if self._are_nearby_parallel_boxes(box_list[i], box_list[j]):
                    a, b = box_list[i], box_list[j]
                    merged_points = np.vstack([a.points, b.points])
                    merged_box = RotatedBox.from_points(merged_points, self.box_type)
                    if merged_box.width / merged_box.height >= self.min_box_aspect:
                        box_list.remove(a)
                        box_list.remove(b)
                        box_list.append(merged_box)
                        return True
        return False

    def _merge_boxes(self, box_list):
        while self._merge_any_two_boxes(box_list):
            pass
        return box_list

    def _fixup_boxes(self, box_list):
        for box in box_list:
            if abs(abs(box.angle) - np.pi / 2) <= 0.01:
                box.angle = np.pi / 2
            if abs(box.angle) <= 0.01:
                box.angle = 0.0
        return box_list


class ExtractAllBoxes(object):
    __provides__ = ["rois"]
    __depends__ = ["boxes", "img", "scale_factor"]

    def __call__(self, boxes, img, scale_factor):
        return [box.extract_from_image(img, 1.0 / scale_factor) for box in boxes]


class FindFirstValidMRZ(object):
    __provides__ = ["box_idx", "roi", "text", "mrz"]
    __depends__ = ["boxes", "img", "img_small", "scale_factor", "__data__"]

    def __init__(self, use_original_image=True, extra_cmdline_params=""):
        self.box_to_mrz = BoxToMRZ(use_original_image, extra_cmdline_params=extra_cmdline_params)

    def __call__(self, boxes, img, img_small, scale_factor, data):
        mrzs = []
        data["__debug__mrz"] = []
        for i, b in enumerate(boxes):
            roi, text, mrz = self.box_to_mrz(b, img, img_small, scale_factor)
            data["__debug__mrz"].append((roi, text, mrz))
            if mrz.valid:
                return i, roi, text, mrz
            elif mrz.valid_score > 0:
                mrzs.append((i, roi, text, mrz))
        if not mrzs:
            return None, None, None, None
        mrzs.sort(key=lambda x: x[3].valid_score)
        return mrzs[-1]


class BoxToMRZ(object):
    __provides__ = ["roi", "text", "mrz"]
    __depends__ = ["box", "img", "img_small", "scale_factor"]

    def __init__(self, use_original_image=True, extra_cmdline_params=""):
        self.use_original_image = use_original_image
        self.extra_cmdline_params = extra_cmdline_params

    def __call__(self, box, img, img_small, scale_factor):
        img = img if self.use_original_image else img_small
        scale = 1.0 / scale_factor if self.use_original_image else 1.0
        roi = box.extract_from_image(img, scale)
        text = ocr(roi, extra_cmdline_params=self.extra_cmdline_params)

        if ">>" in text or (">" in text and "<" not in text):
            roi = roi[::-1, ::-1]
            text = ocr(roi, extra_cmdline_params=self.extra_cmdline_params)

        if "<" not in text:
            return roi, text, MRZ.from_ocr(text)

        mrz = MRZ.from_ocr(text)
        mrz.aux["method"] = "direct"

        if not mrz.valid:
            text, mrz = self._try_larger_image(roi, text, mrz)
        if not mrz.valid:
            text, mrz = self._try_larger_image(roi, text, mrz, 1)
        if not mrz.valid:
            text, mrz = self._try_black_tophat(roi, text, mrz)
        return roi, text, mrz

    def _try_larger_image(self, roi, cur_text, cur_mrz, filter_order=3):
        if roi.shape[1] <= 700:
            scale_by = int(1050.0 / roi.shape[1] + 0.5)
            roi_lg = transform.rescale(
                roi,
                scale_by,
                order=filter_order,
                mode="constant",
                channel_axis=None,
                anti_aliasing=True,
            )
            new_text = ocr(roi_lg, extra_cmdline_params=self.extra_cmdline_params)
            new_mrz = MRZ.from_ocr(new_text)
            new_mrz.aux["method"] = "rescaled(%d)" % filter_order
            if new_mrz.valid_score > cur_mrz.valid_score:
                cur_mrz = new_mrz
                cur_text = new_text
        return cur_text, cur_mrz

    def _try_black_tophat(self, roi, cur_text, cur_mrz):
        roi_b = morphology.black_tophat(roi, morphology.disk(5))
        new_text = ocr(roi_b, extra_cmdline_params=self.extra_cmdline_params)
        new_mrz = MRZ.from_ocr(new_text)
        if new_mrz.valid_score > cur_mrz.valid_score:
            new_mrz.aux["method"] = "black_tophat"
            cur_text, cur_mrz = new_text, new_mrz
        new_text, new_mrz = self._try_larger_image(roi_b, cur_text, cur_mrz)
        if new_mrz.valid_score > cur_mrz.valid_score:
            new_mrz.aux["method"] = "black_tophat(rescaled(3))"
            cur_text, cur_mrz = new_text, new_mrz
        return cur_text, cur_mrz


class TryOtherMaxWidth(object):
    __provides__ = ["mrz_final"]
    __depends__ = ["mrz", "__pipeline__"]

    def __init__(self, other_max_width=1000):
        self.other_max_width = other_max_width

    def __call__(self, mrz, __pipeline__):
        if mrz is None and (__pipeline__["img_binary"].mean() < 0.01 or __pipeline__["img"].mean() > 0.95):
            __pipeline__.replace_component("scaler", Scaler(self.other_max_width))
            new_mrz = __pipeline__["mrz"]
            if new_mrz is not None:
                new_mrz.aux["method"] = new_mrz.aux["method"] + "|max_width(%d)" % self.other_max_width
            mrz = new_mrz
        return mrz


class MRZPipeline(Pipeline):
    def __init__(self, file, extra_cmdline_params=""):
        super(MRZPipeline, self).__init__()
        self.version = "1.0"
        self.file = file
        self.add_component("loader", Loader(file))
        self.add_component("scaler", Scaler())
        self.add_component("boone", BooneTransform())
        self.add_component("box_locator", MRZBoxLocator())
        self.add_component("mrz", FindFirstValidMRZ(extra_cmdline_params=extra_cmdline_params))
        self.add_component("other_max_width", TryOtherMaxWidth())
        self.add_component("extractor", ExtractAllBoxes())

    @property
    def result(self):
        return self["mrz_final"]


def read_mrz(file, save_roi=False, extra_cmdline_params=""):
    p = MRZPipeline(file, extra_cmdline_params)
    mrz = p.result
    if mrz is not None and save_roi:
        mrz.aux["roi"] = p["roi"]
    return mrz

