"""
///////////////////////////////////////////////////
/// VmbPy Asynchronous Grab with OpenCV Example ///
///////////////////////////////////////////////////
"""

import sys
import os
import time
from datetime import datetime
from typing import Optional
from queue import Queue, Empty, Full

import cv2
import numpy as np
from scipy.optimize import minimize
from vmbpy import *


# ============================================================
# INSTELLINGEN
# ============================================================

opencv_display_format = PixelFormat.Bgr8

OUTPUT_DIR = 'recordings'
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
DEFAULT_FPS = 60.0

ROI_WIDTH: Optional[int] = 800
ROI_HEIGHT: Optional[int] = 550
ROI_OFFSET_X: Optional[int] = 310
ROI_OFFSET_Y: Optional[int] = 285

AUTO_MAX_FRAMERATE = True


# ============================================================
# FISHEYE CORRECTIE KLASSE
# ============================================================

class FisheyeCorrector:
    """
    Berekent en past fisheye-correctie toe op basis van
    de cirkels op het air hockey speelveld.

    Het speelveld heeft 5 cirkels:
        - 1 grote in het midden (rood/oranje)
        - 4 kleinere in de hoeken (zwart)

    In werkelijkheid zijn dit perfecte cirkels.
    Door de fisheye lens zien ze er als ellipsen uit.
    We gebruiken dit om k1 en k2 te berekenen.
    """

    def __init__(self):
        self.calibrated = False
        self.camera_matrix = None
        self.dist_coeffs = None
        self.new_camera_matrix = None
        self.map1 = None  # Remap tabel X
        self.map2 = None  # Remap tabel Y
        self.img_size = None

    # ----------------------------------------------------------
    # Stap 1: Detecteer ellipsen in het frame
    # ----------------------------------------------------------

    def detect_ellipses(self, frame: np.ndarray) -> list:
        """
        Detecteer ellipsen (vervormd cirkels) in het frame.
        Geeft lijst van cv2 ellipse tuples terug.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Contrast verbeteren
        gray = cv2.equalizeHist(gray)

        # Ruis verminderen
        blurred = cv2.GaussianBlur(gray, (7, 7), 2)

        # Randen detecteren
        edges = cv2.Canny(blurred, 20, 80)

        # Morfologisch sluiten: kleine gaten in randen dichten
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Contouren zoeken
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE
        )

        ellipses = []
        h, w = frame.shape[:2]

        for cnt in contours:
            if len(cnt) < 5:
                continue

            area = cv2.contourArea(cnt)

            # Filter op oppervlakte (niet te klein, niet te groot)
            # Aanpassen als jouw cirkels groter/kleiner zijn
            if not (500 < area < 60000):
                continue

            # Fit een ellips aan het contour
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (axis_a, axis_b), angle = ellipse

            # Sla randen van het beeld over (vaak veel ruis)
            margin = 20
            if cx < margin or cx > w - margin:
                continue
            if cy < margin or cy > h - margin:
                continue

            # Bereken hoe 'rond' de ellips is
            # 1.0 = perfecte cirkel, 0.0 = lijnstuk
            if max(axis_a, axis_b) == 0:
                continue

            roundness = min(axis_a, axis_b) / max(axis_a, axis_b)

            # We willen vervormd maar nog herkenbaar als cirkel
            # Bij sterke fisheye kunnen hoek-cirkels ratio ~0.5 hebben
            if not (0.35 < roundness < 0.98):
                continue

            ellipses.append({
                'ellipse': ellipse,
                'center': (cx, cy),
                'axes': (axis_a, axis_b),
                'roundness': roundness,
                'area': area
            })

        # Sorteer op oppervlakte (grootste eerst = middelste cirkel)
        ellipses.sort(key=lambda e: e['area'], reverse=True)

        return ellipses

    # ----------------------------------------------------------
    # Stap 2: Bereken hoe 'cirkelachtig' een gecorrigeerd frame is
    # ----------------------------------------------------------

    def _cirkelheid_score(
        self,
        params: list,
        frame: np.ndarray
    ) -> float:
        """
        Doelfunctie voor de optimalisatie.
        Lagere score = ellipsen lijken meer op cirkels na correctie.
        """
        k1, k2 = params
        h, w = frame.shape[:2]

        # Tijdelijke camera matrix
        fx = fy = w * 0.9
        cx_cam, cy_cam = w / 2, h / 2
        cam_mat = np.array([
            [fx,  0, cx_cam],
            [ 0, fy, cy_cam],
            [ 0,  0,      1]
        ], dtype=np.float64)

        dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)

        new_mat, _ = cv2.getOptimalNewCameraMatrix(
            cam_mat, dist, (w, h), 1, (w, h)
        )

        corrected = cv2.undistort(frame, cam_mat, dist, None, new_mat)

        # Detecteer ellipsen in gecorrigeerd frame
        ellipses = self.detect_ellipses(corrected)

        if len(ellipses) < 2:
            # Straf: te weinig cirkels gevonden
            return 10.0

        # Score = gemiddelde afwijking van perfecte cirkel
        scores = []
        for e in ellipses:
            roundness = e['roundness']
            scores.append((1.0 - roundness) ** 2)

        return float(np.mean(scores))

    # ----------------------------------------------------------
    # Stap 3: Kalibreer op basis van een frame
    # ----------------------------------------------------------

    def calibrate(self, frame: np.ndarray) -> bool:
        """
        Analyseer het frame en bereken optimale distortie parameters.
        Geeft True terug als kalibratie geslaagd is.
        """
        h, w = frame.shape[:2]
        self.img_size = (w, h)

        print("\n[KALIBRATIE] Ellipsen zoeken in referentieframe...")
        ellipses = self.detect_ellipses(frame)
        print(f"[KALIBRATIE] {len(ellipses)} ellips(en) gevonden.")

        if len(ellipses) < 2:
            print("[KALIBRATIE] Te weinig ellipsen. Kalibratie mislukt.")
            return False

        for i, e in enumerate(ellipses):
            cx, cy = e['center']
            a, b = e['axes']
            r = e['roundness']
            print(
                f"  Ellips {i+1}: "
                f"centrum=({cx:.0f},{cy:.0f}), "
                f"assen=({a:.0f},{b:.0f}), "
                f"ronding={r:.3f}"
            )

        print("[KALIBRATIE] Parameters optimaliseren...")

        # Startwaarden: negatief k1 voor barrel/fisheye distortie
        result = minimize(
            self._cirkelheid_score,
            x0=[-0.35, 0.05],
            args=(frame,),
            method='Nelder-Mead',
            options={
                'xatol': 1e-4,
                'fatol': 1e-5,
                'maxiter': 300
            }
        )

        k1_opt, k2_opt = result.x
        print(f"[KALIBRATIE] Optimale k1={k1_opt:.5f}, k2={k2_opt:.5f}")
        print(f"[KALIBRATIE] Score: {result.fun:.6f}")

        # Camera matrix opbouwen
        fx = fy = w * 0.9
        cx_cam, cy_cam = w / 2, h / 2

        self.camera_matrix = np.array([
            [fx,  0, cx_cam],
            [ 0, fy, cy_cam],
            [ 0,  0,      1]
        ], dtype=np.float64)

        self.dist_coeffs = np.array(
            [k1_opt, k2_opt, 0.0, 0.0, 0.0],
            dtype=np.float64
        )

        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (w, h),
            1,
            (w, h)
        )

        # Remap tabellen voorberekenen voor snelle correctie
        # (veel sneller dan undistort() elke frame aanroepen)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.new_camera_matrix,
            (w, h),
            cv2.CV_16SC2
        )

        self.calibrated = True
        print("[KALIBRATIE] Succesvol afgerond!\n")
        return True

    # ----------------------------------------------------------
    # Stap 4: Pas correctie toe op een frame
    # ----------------------------------------------------------

    def correct(self, frame: np.ndarray) -> np.ndarray:
        """
        Pas fisheye-correctie toe op een frame.
        Geeft gecorrigeerd frame terug, of origineel als niet gekalibreerd.
        """
        if not self.calibrated:
            return frame

        # remap is sneller dan undistort() per frame
        corrected = cv2.remap(
            frame,
            self.map1,
            self.map2,
            cv2.INTER_LINEAR
        )

        return corrected

    # ----------------------------------------------------------
    # Debug: teken gevonden ellipsen op frame
    # ----------------------------------------------------------

    def draw_ellipses(self, frame: np.ndarray) -> np.ndarray:
        """Teken gevonden ellipsen op het frame (voor debug)."""
        debug = frame.copy()
        ellipses = self.detect_ellipses(frame)

        for i, e in enumerate(ellipses):
            cv2.ellipse(debug, e['ellipse'], (0, 255, 0), 2)
            cx, cy = int(e['center'][0]), int(e['center'][1])
            cv2.circle(debug, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(
                debug,
                f"{e['roundness']:.2f}",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1
            )

        return debug


# ============================================================
# STARTSCHERM
# ============================================================

def print_preamble():
    print('///////////////////////////////////////////////////')
    print('/// VmbPy Asynchronous Grab with OpenCV Example ///')
    print('///////////////////////////////////////////////////\n')


# ============================================================
# FOUTAFHANDELING
# ============================================================

def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')
    if usage:
        print_usage()
    sys.exit(return_code)


# ============================================================
# GEBRUIKSINFORMATIE
# ============================================================

def print_usage():
    print('Usage:')
    print('    python asynchronous_grab_opencv_record.py [camera_id]')
    print('    python asynchronous_grab_opencv_record.py [/h] [-h]')
    print()
    print('Parameters:')
    print('    camera_id   ID of the camera to use '
          '(using first camera if not specified)')
    print()


# ============================================================
# COMMAND LINE ARGUMENTEN
# ============================================================

def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    argc = len(args)

    for arg in args:
        if arg in ('/h', '-h'):
            print_usage()
            sys.exit(0)

    if argc > 1:
        abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

    return None if argc == 0 else args[0]


# ============================================================
# CAMERA OPENEN
# ============================================================

def get_camera(camera_id: Optional[str]) -> Camera:
    with VmbSystem.get_instance() as vmb:
        if camera_id:
            try:
                return vmb.get_camera_by_id(camera_id)
            except VmbCameraError:
                abort(f"Failed to access Camera '{camera_id}'. Abort.")
        else:
            cams = vmb.get_all_cameras()
            if not cams:
                abort('No Cameras accessible. Abort.')
            print(f"Camera gevonden: {cams[0].get_name()}")
            return cams[0]


# ============================================================
# CAMERA INSTELLINGEN (ongewijzigd)
# ============================================================

def setup_camera(cam: Camera):
    with cam:
        try:
            cam.ExposureAuto.set('Continuous')
            print('Auto exposure: Continuous')
        except (AttributeError, VmbFeatureError):
            print('Auto exposure niet beschikbaar.')

        try:
            cam.BalanceWhiteAuto.set('Continuous')
            print('Auto white balance: Continuous')
        except (AttributeError, VmbFeatureError):
            print('Auto white balance niet beschikbaar.')

        try:
            stream = cam.get_streams()[0]
            stream.GVSPAdjustPacketSize.run()
            while not stream.GVSPAdjustPacketSize.is_done():
                pass
            print('GigE packet size automatisch aangepast.')
        except (AttributeError, VmbFeatureError):
            print('Kon GigE packet size niet automatisch aanpassen.')


def _round_to_increment(value: int, increment: int) -> int:
    if increment <= 1:
        return value
    return (value // increment) * increment


def setup_roi(cam: Camera):
    with cam:
        try:
            cam.OffsetX.set(0)
            cam.OffsetY.set(0)

            if ROI_WIDTH is not None:
                cam.Width.set(_round_to_increment(ROI_WIDTH, cam.Width.get_increment()))

            if ROI_HEIGHT is not None:
                cam.Height.set(_round_to_increment(ROI_HEIGHT, cam.Height.get_increment()))

            if ROI_OFFSET_X is not None:
                cam.OffsetX.set(_round_to_increment(ROI_OFFSET_X, cam.OffsetX.get_increment()))

            if ROI_OFFSET_Y is not None:
                cam.OffsetY.set(_round_to_increment(ROI_OFFSET_Y, cam.OffsetY.get_increment()))

            print(f"ROI ingesteld op {cam.Width.get()}x{cam.Height.get()} "
                  f"(offset {cam.OffsetX.get()}, {cam.OffsetY.get()})")

        except (AttributeError, VmbFeatureError) as e:
            print(f"Kon ROI niet (volledig) instellen: {e}")

        try:
            _, max_fps = cam.AcquisitionFrameRate.get_range()
            print(f"Maximaal haalbare framerate: {max_fps:.1f} fps")
            if AUTO_MAX_FRAMERATE:
                cam.AcquisitionFrameRate.set(max_fps)
                print(f"Framerate ingesteld op maximum: {max_fps:.1f} fps")
        except (AttributeError, VmbFeatureError) as e:
            print(f"Kon framerate niet ophalen/instellen: {e}")


def setup_pixel_format(cam: Camera):
    cam_formats = cam.get_pixel_formats()
    cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
    convertible_color_formats = tuple(
        f for f in cam_color_formats
        if opencv_display_format in f.get_convertible_formats()
    )
    cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
    convertible_mono_formats = tuple(
        f for f in cam_mono_formats
        if opencv_display_format in f.get_convertible_formats()
    )

    if opencv_display_format in cam_formats:
        cam.set_pixel_format(opencv_display_format)
        print('Pixel format ingesteld op Bgr8.')
    elif convertible_color_formats:
        cam.set_pixel_format(convertible_color_formats[0])
        print(f'Camera pixel format: {convertible_color_formats[0]}')
    elif convertible_mono_formats:
        cam.set_pixel_format(convertible_mono_formats[0])
        print(f'Camera pixel format: {convertible_mono_formats[0]}')
    else:
        abort('Camera does not support an OpenCV compatible format. Abort.')


def get_camera_fps(cam: Camera) -> float:
    try:
        return float(cam.AcquisitionFrameRate.get())
    except (AttributeError, VmbFeatureError):
        return DEFAULT_FPS


def create_video_writer(cam: Camera) -> 'tuple[cv2.VideoWriter, str]':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(OUTPUT_DIR, f'opname_{timestamp}.mp4')
    width = cam.Width.get()
    height = cam.Height.get()
    fps = get_camera_fps(cam)
    print(f'Video resolutie: {width}x{height}')
    print(f'Video FPS: {fps:.1f}')
    writer = cv2.VideoWriter(filename, FOURCC, fps, (width, height))
    if not writer.isOpened():
        abort('Kon VideoWriter niet openen.')
    return writer, filename


# ============================================================
# FRAME HANDLER
# ============================================================

class Handler:

    def __init__(self):
        self.display_queue = Queue(10)
        self.frame_count = 0
        self.last_frame_time = None
        self.fps = 0.0
        self._last_print_time = 0.0

    def get_image(self, timeout: Optional[float] = None):
        return self.display_queue.get(True, timeout=timeout)

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        status = frame.get_status()

        if status == FrameStatus.Complete:
            self.frame_count += 1
            current_time = time.perf_counter()

            if self.last_frame_time is not None:
                delta = current_time - self.last_frame_time
                if delta > 0:
                    self.fps = 1.0 / delta

            self.last_frame_time = current_time

            if current_time - self._last_print_time > 0.5:
                print(
                    f"Frame ID: {self.frame_count} | FPS: {self.fps:.1f}",
                    flush=True
                )
                self._last_print_time = current_time

            display = (
                frame
                if frame.get_pixel_format() == opencv_display_format
                else frame.convert_pixel_format(opencv_display_format)
            )

            image = display.as_opencv_image()

            try:
                self.display_queue.put_nowait(image)
            except Full:
                try:
                    self.display_queue.get_nowait()
                except Empty:
                    pass
                try:
                    self.display_queue.put_nowait(image)
                except Full:
                    pass

        else:
            print(f"Onvolledig frame (status: {status})", flush=True)

        cam.queue_frame(frame)


# ============================================================
# MAIN
# ============================================================

def main():
    print_preamble()
    cam_id = parse_args()

    # Fisheye corrector aanmaken
    corrector = FisheyeCorrector()

    with VmbSystem.get_instance():
        with get_camera(cam_id) as cam:
            print(f"Camera geopend: {cam.get_name()}")

            setup_camera(cam)
            setup_roi(cam)
            setup_pixel_format(cam)

            handler = Handler()
            writer, filename = create_video_writer(cam)

            print(f"Opname wordt weggeschreven naar: {filename}\n")

            try:
                cam.start_streaming(handler=handler, buffer_count=10)

                print("Streaming gestart, wachten op frames...")
                print("Toetsen:")
                print("  [K] - Kalibreer fisheye correctie op dit frame")
                print("  [D] - Debug: toon gevonden ellipsen")
                print("  [C] - Schakel correctie aan/uit")
                print("  [Enter/ESC] - Stoppen\n")

                window_name = "Mako camera"

                # Status
                show_corrected = True   # Correctie tonen of niet
                debug_mode = False       # Ellipsen tekenen

                while True:
                    try:
                        raw_frame = handler.get_image(timeout=0.5)
                    except Empty:
                        key = cv2.waitKey(1) & 0xFF
                        if key in (13, 27):
                            break
                        continue

                    # ----------------------------------------
                    # Fisheye correctie toepassen
                    # ----------------------------------------

                    if corrector.calibrated and show_corrected:
                        display = corrector.correct(raw_frame)
                    else:
                        display = raw_frame

                    # ----------------------------------------
                    # Debug overlay
                    # ----------------------------------------

                    if debug_mode:
                        # Teken ellipsen op het ORIGINELE frame
                        display = corrector.draw_ellipses(raw_frame)

                    # ----------------------------------------
                    # Status tekst
                    # ----------------------------------------

                    status_text = (
                        "Gecorrigeerd" if (corrector.calibrated and show_corrected)
                        else "NIET gecorrigeerd"
                    )

                    cv2.putText(
                        display,
                        f"[K]=Kalibreer [D]=Debug [C]=Toggle | {status_text}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0) if corrector.calibrated else (0, 0, 255),
                        2
                    )

                    # ----------------------------------------
                    # Tonen en opslaan
                    # ----------------------------------------

                    cv2.imshow(window_name, display)
                    writer.write(display)

                    # ----------------------------------------
                    # Toetsen
                    # ----------------------------------------

                    key = cv2.waitKey(1) & 0xFF

                    if key in (13, 27):
                        # Enter of ESC → stoppen
                        break

                    elif key == ord('k') or key == ord('K'):
                        # Kalibreer op huidig frame
                        print("[KALIBRATIE] Kalibratie gestart op huidig frame...")
                        success = corrector.calibrate(raw_frame)
                        if success:
                            # Sla referentieframe op voor inspectie
                            cv2.imwrite('calibratie_referentie.jpg', raw_frame)
                            cv2.imwrite(
                                'calibratie_gecorrigeerd.jpg',
                                corrector.correct(raw_frame)
                            )
                            print("[KALIBRATIE] Referentieframes opgeslagen.")
                        show_corrected = True

                    elif key == ord('c') or key == ord('C'):
                        # Schakel correctie aan/uit
                        show_corrected = not show_corrected
                        staat = "AAN" if show_corrected else "UIT"
                        print(f"[CORRECTIE] Fisheye correctie: {staat}")

                    elif key == ord('d') or key == ord('D'):
                        # Debug modus
                        debug_mode = not debug_mode
                        staat = "AAN" if debug_mode else "UIT"
                        print(f"[DEBUG] Ellips-detectie overlay: {staat}")

                    # Venster gesloten?
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        break

            finally:
                cam.stop_streaming()
                writer.release()
                cv2.destroyAllWindows()

                print("\nVideo-opname afgesloten en opgeslagen.")
                print(f"Totaal ontvangen frames: {handler.frame_count}")


# ============================================================
# PROGRAMMA STARTEN
# ============================================================

if __name__ == '__main__':
    main()