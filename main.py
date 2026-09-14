"""
Smart Document Scanner — Android app (Kivy).

Reuses the exact same OpenCV processing engine as the desktop version
(scanner_core/), with a touch-first mobile UI: take/choose a photo,
drag corners, pick a processing mode, compare before/after, save.
"""
import os
import traceback

os.environ.setdefault("KIVY_NO_ARGS", "1")

import numpy as np
import cv2

from kivy.app import App
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.properties import NumericProperty, BooleanProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.image import Image as KivyImage
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.stencilview import StencilView
from kivy.clock import Clock
from kivy.metrics import dp

from scanner_core import config
from scanner_core.utils import load_image_unicode_safe, save_image_unicode_safe
from scanner_core.detection.document_detector import detect_document
from scanner_core.detection.corner_detector import refine_corners
from scanner_core.processing.pipeline import process_document
from scanner_core.quality.quality_control import run_quality_control
from scanner_core.export.exporter import export as export_document

try:
    from plyer import filechooser as plyer_filechooser
except Exception:  # pragma: no cover
    plyer_filechooser = None

try:
    from android.permissions import request_permissions, Permission  # noqa
    ANDROID = True
except Exception:
    ANDROID = False


APP_PICTURES_DIR = os.path.join(os.path.expanduser("~"), "SmartDocumentScanner")
os.makedirs(APP_PICTURES_DIR, exist_ok=True)

# The font Kivy ships with by default (Roboto) has no Arabic glyphs, which
# renders the whole UI as empty boxes. We bundle a proper Arabic-capable
# font and register it AS "Roboto" so every widget (Button, Label, Spinner)
# picks it up automatically without needing font_name set everywhere.
_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts", "NotoNaskhArabic-Regular.ttf")
if os.path.exists(_FONT_PATH):
    LabelBase.register(name="Roboto", fn_regular=_FONT_PATH)


# ---------------------------------------------------------------------------
# Helpers: OpenCV BGR array <-> Kivy Texture
# ---------------------------------------------------------------------------
def cv_to_texture(bgr_image: np.ndarray) -> Texture:
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    texture = Texture.create(size=(w, h), colorfmt="rgb")
    texture.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
    texture.flip_vertical()
    return texture


# ---------------------------------------------------------------------------
# Corner editing overlay
# ---------------------------------------------------------------------------
class CornerOverlay(Widget):
    """Sits on top of a KivyImage showing the loaded photo; lets the user
    drag 4 points to correct the detected document quad. Coordinates are
    kept internally in ORIGINAL image pixel space and mapped to/from the
    widget's on-screen displayed rectangle (which changes with orientation
    and screen size)."""

    HANDLE_R = dp(14)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_widget: KivyImage | None = None
        self.orig_w = 1
        self.orig_h = 1
        self.points = [[0, 0], [1, 0], [1, 1], [0, 1]]  # image-space
        self._drag_index = None
        self.bind(pos=self._redraw, size=self._redraw)

    def attach(self, image_widget: KivyImage, orig_w: int, orig_h: int, quad):
        self.image_widget = image_widget
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.set_points(quad)

    def set_points(self, quad):
        self.points = [[float(p[0]), float(p[1])] for p in quad]
        self._redraw()

    def get_points_image_space(self):
        return np.array(self.points, dtype="float32")

    # -- coordinate mapping -------------------------------------------------
    def _display_rect(self):
        """Returns (x, y, w, h) of the actually displayed image within the
        widget, accounting for aspect-fit letterboxing."""
        if self.image_widget is None or not self.image_widget.texture:
            return (self.x, self.y, self.width, self.height)
        tw, th = self.image_widget.texture.size
        if tw == 0 or th == 0:
            return (self.x, self.y, self.width, self.height)
        scale = min(self.width / tw, self.height / th)
        dw, dh = tw * scale, th * scale
        dx = self.x + (self.width - dw) / 2.0
        dy = self.y + (self.height - dh) / 2.0
        return (dx, dy, dw, dh)

    def _img_to_widget(self, pt):
        rx, ry, rw, rh = self._display_rect()
        ix, iy = pt
        wx = rx + (ix / self.orig_w) * rw
        wy = ry + rh - (iy / self.orig_h) * rh
        return wx, wy

    def _widget_to_img(self, wx, wy):
        rx, ry, rw, rh = self._display_rect()
        if rw == 0 or rh == 0:
            return 0, 0
        ix = (wx - rx) / rw * self.orig_w
        iy = (ry + rh - wy) / rh * self.orig_h
        ix = min(max(ix, 0), self.orig_w - 1)
        iy = min(max(iy, 0), self.orig_h - 1)
        return ix, iy

    # -- touch handling -------------------------------------------------
    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        for i, p in enumerate(self.points):
            wx, wy = self._img_to_widget(p)
            if (touch.x - wx) ** 2 + (touch.y - wy) ** 2 <= (self.HANDLE_R * 2.2) ** 2:
                self._drag_index = i
                touch.grab(self)
                return True
        return False

    def on_touch_move(self, touch):
        if touch.grab_current is self and self._drag_index is not None:
            ix, iy = self._widget_to_img(touch.x, touch.y)
            self.points[self._drag_index] = [ix, iy]
            self._redraw()
            return True
        return False

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            self._drag_index = None
            return True
        return False

    def _redraw(self, *_args):
        self.canvas.clear()
        from kivy.graphics import Color, Ellipse, Line

        with self.canvas:
            Color(0.24, 0.86, 0.35, 1)
            widget_pts = [self._img_to_widget(p) for p in self.points]
            flat = []
            for wx, wy in widget_pts:
                flat += [wx, wy]
            flat += [widget_pts[0][0], widget_pts[0][1]]
            Line(points=flat, width=dp(2), close=True)

            for wx, wy in widget_pts:
                Color(0.95, 0.25, 0.25, 1)
                Ellipse(pos=(wx - self.HANDLE_R, wy - self.HANDLE_R), size=(self.HANDLE_R * 2, self.HANDLE_R * 2))
                Color(1, 1, 1, 1)
                Line(circle=(wx, wy, self.HANDLE_R), width=dp(1.5))


# ---------------------------------------------------------------------------
# Before / After compare widget
# ---------------------------------------------------------------------------
class BeforeAfterView(FloatLayout):
    ratio = NumericProperty(0.5)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.before_img = KivyImage(allow_stretch=True, keep_ratio=True, size_hint=(1, 1), pos_hint={"x": 0, "y": 0})
        self.add_widget(self.before_img)

        self.stencil = StencilView(size_hint=(None, 1), pos_hint={"y": 0})
        self.after_img = KivyImage(allow_stretch=True, keep_ratio=True, size_hint=(None, None))
        self.stencil.add_widget(self.after_img)
        self.add_widget(self.stencil)

        self.bind(size=self._layout, pos=self._layout, ratio=self._layout)

    def set_images(self, before_bgr, after_bgr):
        self.before_img.texture = cv_to_texture(before_bgr)
        self.after_img.texture = cv_to_texture(after_bgr)
        self._layout()

    def _layout(self, *_args):
        x, y = self.pos
        w, h = self.size
        divider_x = x + w * self.ratio
        self.stencil.pos = (divider_x, y)
        self.stencil.size = (max(0.0, x + w - divider_x), h)
        # after_img must cover the FULL widget rect (so it lines up with
        # before_img) — the stencil clips whatever falls outside its rect.
        self.after_img.pos = (x, y)
        self.after_img.size = (w, h)


# ---------------------------------------------------------------------------
# Simple file chooser popup (used as a desktop-testing fallback, and as a
# backup path on Android if plyer's native chooser is unavailable)
# ---------------------------------------------------------------------------
def show_filechooser_popup(on_selected):
    layout = BoxLayout(orientation="vertical")
    chooser = FileChooserIconView(filters=["*.png", "*.jpg", "*.jpeg", "*.bmp"])
    layout.add_widget(chooser)
    btns = BoxLayout(size_hint_y=None, height=dp(48))
    popup = Popup(title="اختر صورة", content=layout, size_hint=(0.9, 0.9))

    def choose(_):
        if chooser.selection:
            popup.dismiss()
            on_selected(chooser.selection[0])

    ok_btn = Button(text="اختيار")
    ok_btn.bind(on_release=choose)
    cancel_btn = Button(text="إلغاء")
    cancel_btn.bind(on_release=popup.dismiss)
    btns.add_widget(cancel_btn)
    btns.add_widget(ok_btn)
    layout.add_widget(btns)
    popup.open()


KV = """
<RootWidget>:
    orientation: "vertical"

    BoxLayout:
        size_hint_y: None
        height: dp(48)
        padding: dp(4)
        spacing: dp(4)
        Button:
            text: "إضافة صورة"
            on_release: root.on_add_image()
        Button:
            text: "اكتشاف"
            on_release: root.on_detect()
        Button:
            text: "معالجة"
            on_release: root.on_process()
        Button:
            text: "إعادة ضبط"
            size_hint_x: 0.7
            font_size: dp(12)
            on_release: root.on_reset()

    BoxLayout:
        size_hint_y: None
        height: dp(44)
        padding: dp(4)
        spacing: dp(4)
        Spinner:
            id: mode_spinner
            text: "Document"
            values: ["Original", "Color", "Clean", "Document", "Black & White"]
            on_text: root.on_mode_changed(self.text)
        Button:
            text: "حفظ"
            size_hint_x: 0.5
            on_release: root.on_save()

    FloatLayout:
        id: view_container

    Label:
        id: status_label
        text: root.status_text
        size_hint_y: None
        height: dp(28)
        font_size: dp(12)
        color: 0.9, 0.6, 0.2, 1
"""


class RootWidget(BoxLayout):
    status_text = StringProperty("مرحبًا! أضف صورة وثيقة للبدء.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.original_image = None
        self.detection = None
        self.current_quad = None
        self.pipeline_result = None
        self.current_path = None

        self.mode_view = "empty"  # empty | corners | compare
        self.image_widget = None
        self.corner_overlay = None
        self.compare_view = None

        Clock.schedule_once(lambda *_: self._show_empty(), 0)

    # ---------------- view switching ----------------
    def _clear_container(self):
        self.ids.view_container.clear_widgets()

    def _show_empty(self):
        self._clear_container()
        box = BoxLayout(orientation="vertical")
        box.add_widget(Label(text="اضغط \u201c📷 صورة\u201d لاختيار أو تصوير وثيقة", font_size=dp(16)))
        self.ids.view_container.add_widget(box)
        self.mode_view = "empty"

    def _show_corners(self):
        self._clear_container()
        container = FloatLayout()
        self.image_widget = KivyImage(allow_stretch=True, keep_ratio=True)
        self.image_widget.texture = cv_to_texture(self.original_image)
        container.add_widget(self.image_widget)

        self.corner_overlay = CornerOverlay()
        h, w = self.original_image.shape[:2]
        self.corner_overlay.attach(self.image_widget, w, h, self.current_quad)
        container.add_widget(self.corner_overlay)

        self.ids.view_container.add_widget(container)
        self.mode_view = "corners"

    def _show_compare(self):
        self._clear_container()
        self.compare_view = BeforeAfterView()
        self.compare_view.set_images(self.pipeline_result.warped_original, self.pipeline_result.final)

        from kivy.uix.slider import Slider

        wrapper = BoxLayout(orientation="vertical")
        wrapper.add_widget(self.compare_view)
        slider = Slider(min=0, max=1, value=0.5, size_hint_y=None, height=dp(36))
        slider.bind(value=lambda _s, v: setattr(self.compare_view, "ratio", v))
        wrapper.add_widget(slider)
        self.ids.view_container.add_widget(wrapper)
        self.mode_view = "compare"

    # ---------------- actions ----------------
    def on_add_image(self):
        if plyer_filechooser is not None:
            try:
                plyer_filechooser.open_file(on_selection=self._on_file_selected, filters=[["Images", "*.jpg", "*.jpeg", "*.png"]])
                return
            except Exception:
                traceback.print_exc()
        show_filechooser_popup(lambda path: self._on_file_selected([path]))

    def _on_file_selected(self, selection):
        if not selection:
            return
        path = selection[0]
        img = load_image_unicode_safe(path)
        if img is None:
            self.status_text = "تعذر فتح الصورة المحددة."
            return
        self.current_path = path
        self.original_image = img
        self.detection = None
        self.pipeline_result = None
        self.status_text = f"تم تحميل: {os.path.basename(path)}"
        self.on_detect()

    def on_detect(self):
        if self.original_image is None:
            self.status_text = "أضف صورة أولًا."
            return
        try:
            self.detection = detect_document(self.original_image)
            self.current_quad = refine_corners(self.original_image, self.detection.quad)
        except Exception as exc:
            traceback.print_exc()
            self.status_text = f"خطأ أثناء الاكتشاف: {exc}"
            return

        self._show_corners()
        if not self.detection.success or self.detection.confidence < 0.4:
            self.status_text = "ثقة منخفضة — اسحب النقاط الحمراء لتصحيح الزوايا ثم اضغط معالجة."
        else:
            self.status_text = f"تم الاكتشاف (ثقة {self.detection.confidence:.0%}). راجع الزوايا ثم اضغط معالجة."

    def on_mode_changed(self, _text):
        if self.pipeline_result is not None:
            self.on_process()

    def on_process(self):
        if self.original_image is None:
            self.status_text = "أضف صورة أولًا."
            return
        if self.mode_view == "corners" and self.corner_overlay is not None:
            self.current_quad = self.corner_overlay.get_points_image_space()
        if self.current_quad is None:
            self.status_text = "اضغط اكتشاف أولًا."
            return

        mode = self.ids.mode_spinner.text
        try:
            result = process_document(self.original_image, self.current_quad, mode=mode)
            self.pipeline_result = result
            qc = run_quality_control(
                original_image=self.original_image,
                quad=self.current_quad,
                detection_confidence=self.detection.confidence if self.detection else 0.0,
                detection_success=self.detection.success if self.detection else False,
                final_image=result.final,
            )
        except Exception as exc:
            traceback.print_exc()
            self.status_text = f"خطأ أثناء المعالجة: {exc}"
            return

        self._show_compare()
        self.status_text = "تمت المعالجة بنجاح." if qc.ok else ("تنبيه: " + " | ".join(qc.warnings[:2]))

    def on_reset(self):
        self.original_image = None
        self.detection = None
        self.current_quad = None
        self.pipeline_result = None
        self.current_path = None
        self._show_empty()
        self.status_text = "تمت إعادة الضبط."

    def on_save(self):
        if self.pipeline_result is None:
            self.status_text = "لا توجد نتيجة لحفظها بعد."
            return
        base = "scan"
        if self.current_path:
            base = os.path.splitext(os.path.basename(self.current_path))[0] + "_scanned"
        out_path = os.path.join(APP_PICTURES_DIR, base + ".pdf")
        n = 1
        while os.path.exists(out_path):
            out_path = os.path.join(APP_PICTURES_DIR, f"{base}_{n}.pdf")
            n += 1
        try:
            export_document(self.pipeline_result.final, out_path)
        except Exception as exc:
            traceback.print_exc()
            self.status_text = f"فشل الحفظ: {exc}"
            return
        self.status_text = f"تم الحفظ: {out_path}"


class ScannerApp(App):
    def build(self):
        Window.clearcolor = (0.11, 0.11, 0.13, 1)
        Builder.load_string(KV)
        if ANDROID:
            try:
                request_permissions([Permission.CAMERA, Permission.READ_MEDIA_IMAGES, Permission.WRITE_EXTERNAL_STORAGE])
            except Exception:
                pass
        return RootWidget()


if __name__ == "__main__":
    ScannerApp().run()
