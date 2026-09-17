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
from vmbpy import *


# ============================================================
# INSTELLINGEN
# ============================================================

# Alle frames worden naar dit formaat geconverteerd voor OpenCV
opencv_display_format = PixelFormat.Bgr8


# ------------------------------------------------------------
# Instellingen voor opname
# ------------------------------------------------------------

OUTPUT_DIR = 'recordings'

# MP4-opname
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

# Fallback als de camera geen framerate-feature heeft
DEFAULT_FPS = 120.0


# ------------------------------------------------------------
# ROI
# ------------------------------------------------------------

ROI_WIDTH: Optional[int] = 800         # minimaal beeld 800 maximaal beeld 1456
ROI_HEIGHT: Optional[int] = 550        # minimaal beeld 550 maximaal beeld 1088

ROI_OFFSET_X: Optional[int] = 310
ROI_OFFSET_Y: Optional[int] = 285


# ------------------------------------------------------------
# Framerate
# ------------------------------------------------------------

# True = maximale framerate van de camera proberen in te stellen
AUTO_MAX_FRAMERATE = True


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
        abort(
            reason="Invalid number of arguments. Abort.",
            return_code=2,
            usage=True
        )

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
                abort(
                    f"Failed to access Camera "
                    f"'{camera_id}'. Abort."
                )

        else:

            cams = vmb.get_all_cameras()

            if not cams:
                abort(
                    'No Cameras accessible. Abort.'
                )

            print(
                f"Camera gevonden: {cams[0].get_name()}"
            )

            return cams[0]


# ============================================================
# ALGEMENE CAMERA-INSTELLINGEN
# ============================================================

def setup_camera(cam: Camera):

    with cam:

        # ----------------------------------------------------
        # Auto exposure
        # ----------------------------------------------------

        try:
            cam.ExposureAuto.set('Continuous')

            print(
                'Auto exposure: Continuous'
            )

        except (AttributeError, VmbFeatureError):

            print(
                'Auto exposure niet beschikbaar.'
            )

        # ----------------------------------------------------
        # Auto white balance
        # ----------------------------------------------------

        try:
            cam.BalanceWhiteAuto.set('Continuous')

            print(
                'Auto white balance: Continuous'
            )

        except (AttributeError, VmbFeatureError):

            print(
                'Auto white balance niet beschikbaar.'
            )

        # ----------------------------------------------------
        # GigE packet size aanpassen
        # ----------------------------------------------------

        try:

            stream = cam.get_streams()[0]

            stream.GVSPAdjustPacketSize.run()

            while not stream.GVSPAdjustPacketSize.is_done():
                pass

            print(
                'GigE packet size automatisch aangepast.'
            )

        except (AttributeError, VmbFeatureError):

            print(
                'Kon GigE packet size niet automatisch aanpassen.'
            )


# ============================================================
# ROI
# ============================================================

def _round_to_increment(
    value: int,
    increment: int
) -> int:

    """
    Rondt een waarde naar beneden af naar
    het dichtstbijzijnde geldige increment.
    """

    if increment <= 1:
        return value

    return (value // increment) * increment


def setup_roi(cam: Camera):

    with cam:

        try:

            # ------------------------------------------------
            # Eerst offsets naar 0
            # ------------------------------------------------

            cam.OffsetX.set(0)
            cam.OffsetY.set(0)

            # ------------------------------------------------
            # Width
            # ------------------------------------------------

            if ROI_WIDTH is not None:

                width = _round_to_increment(
                    ROI_WIDTH,
                    cam.Width.get_increment()
                )

                cam.Width.set(width)

            # ------------------------------------------------
            # Height
            # ------------------------------------------------

            if ROI_HEIGHT is not None:

                height = _round_to_increment(
                    ROI_HEIGHT,
                    cam.Height.get_increment()
                )

                cam.Height.set(height)

            # ------------------------------------------------
            # Offset X
            # ------------------------------------------------

            if ROI_OFFSET_X is not None:

                offset_x = _round_to_increment(
                    ROI_OFFSET_X,
                    cam.OffsetX.get_increment()
                )

                cam.OffsetX.set(offset_x)

            # ------------------------------------------------
            # Offset Y
            # ------------------------------------------------

            if ROI_OFFSET_Y is not None:

                offset_y = _round_to_increment(
                    ROI_OFFSET_Y,
                    cam.OffsetY.get_increment()
                )

                cam.OffsetY.set(offset_y)

            print(
                f"ROI ingesteld op "
                f"{cam.Width.get()}x{cam.Height.get()} "
                f"(offset "
                f"{cam.OffsetX.get()}, "
                f"{cam.OffsetY.get()})"
            )

        except (AttributeError, VmbFeatureError) as e:

            print(
                f"Kon ROI niet (volledig) instellen: {e}"
            )

        # ----------------------------------------------------
        # Framerate controleren
        # ----------------------------------------------------

        try:

            _, max_fps = (
                cam.AcquisitionFrameRate.get_range()
            )

            print(
                f"Maximaal haalbare framerate volgens "
                f"camera: {max_fps:.1f} fps"
            )

            if AUTO_MAX_FRAMERATE:

                cam.AcquisitionFrameRate.set(max_fps)

                print(
                    f"Framerate ingesteld op maximum: "
                    f"{max_fps:.1f} fps"
                )

        except (AttributeError, VmbFeatureError) as e:

            print(
                f"Kon framerate-informatie niet "
                f"ophalen/instellen: {e}"
            )


# ============================================================
# PIXEL FORMAT
# ============================================================

def setup_pixel_format(cam: Camera):

    cam_formats = cam.get_pixel_formats()

    # --------------------------------------------------------
    # Kleurformaten
    # --------------------------------------------------------

    cam_color_formats = intersect_pixel_formats(
        cam_formats,
        COLOR_PIXEL_FORMATS
    )

    convertible_color_formats = tuple(
        f for f in cam_color_formats
        if opencv_display_format
        in f.get_convertible_formats()
    )

    # --------------------------------------------------------
    # Monochrome formaten
    # --------------------------------------------------------

    cam_mono_formats = intersect_pixel_formats(
        cam_formats,
        MONO_PIXEL_FORMATS
    )

    convertible_mono_formats = tuple(
        f for f in cam_mono_formats
        if opencv_display_format
        in f.get_convertible_formats()
    )

    # --------------------------------------------------------
    # Voorkeur: direct BGR8
    # --------------------------------------------------------

    if opencv_display_format in cam_formats:

        cam.set_pixel_format(
            opencv_display_format
        )

        print(
            'Pixel format ingesteld op Bgr8.'
        )

    # --------------------------------------------------------
    # Anders kleurformaat dat naar BGR8 kan
    # --------------------------------------------------------

    elif convertible_color_formats:

        selected_format = (
            convertible_color_formats[0]
        )

        cam.set_pixel_format(
            selected_format
        )

        print(
            f'Camera pixel format: '
            f'{selected_format}'
        )

    # --------------------------------------------------------
    # Anders monochroom formaat
    # --------------------------------------------------------

    elif convertible_mono_formats:

        selected_format = (
            convertible_mono_formats[0]
        )

        cam.set_pixel_format(
            selected_format
        )

        print(
            f'Camera pixel format: '
            f'{selected_format}'
        )

    else:

        abort(
            'Camera does not support an '
            'OpenCV compatible format. Abort.'
        )


# ============================================================
# FRAMERATE VOOR VIDEO
# ============================================================

def get_camera_fps(cam: Camera) -> float:

    try:

        return float(
            cam.AcquisitionFrameRate.get()
        )

    except (AttributeError, VmbFeatureError):

        return DEFAULT_FPS


# ============================================================
# VIDEO WRITER
# ============================================================

def create_video_writer(
    cam: Camera
) -> 'tuple[cv2.VideoWriter, str]':

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        '%Y%m%d_%H%M%S'
    )

    filename = os.path.join(
        OUTPUT_DIR,
        f'opname_{timestamp}.mp4'
    )

    width = cam.Width.get()
    height = cam.Height.get()

    fps = get_camera_fps(cam)

    print(
        f'Video resolutie: {width}x{height}'
    )

    print(
        f'Video FPS: {fps:.1f}'
    )

    writer = cv2.VideoWriter(
        filename,
        FOURCC,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        abort(
            'Kon VideoWriter niet openen. '
            'Probeer FOURCC="XVID" met '
            'bestandsextensie .avi.'
        )

    return writer, filename


# ============================================================
# FRAME HANDLER
# ============================================================

class Handler:

    def __init__(self):

        # Queue waarin afbeeldingen voor OpenCV komen
        self.display_queue = Queue(10)

        # Teller voor frame ID
        self.frame_count = 0

        # Tijdstip van vorig ontvangen frame
        self.last_frame_time = None

        # Gemeten FPS
        self.fps = 0.0

        # Tijdstip van laatste console-print (om spam bij hoge fps te beperken)
        self._last_print_time = 0.0


    def get_image(
        self,
        timeout: Optional[float] = None
    ):

        return self.display_queue.get(
            True,
            timeout=timeout
        )


    def __call__(
        self,
        cam: Camera,
        stream: Stream,
        frame: Frame
    ):

        status = frame.get_status()

        # ----------------------------------------------------
        # Alleen complete frames verwerken
        # ----------------------------------------------------

        if status == FrameStatus.Complete:

            # --------------------------------------------
            # Frame ID verhogen
            # --------------------------------------------

            self.frame_count += 1

            # --------------------------------------------
            # Huidige tijd ophalen
            # --------------------------------------------

            current_time = time.perf_counter()

            # --------------------------------------------
            # FPS berekenen
            # --------------------------------------------

            if self.last_frame_time is not None:

                delta_time = (
                    current_time
                    - self.last_frame_time
                )

                if delta_time > 0:

                    self.fps = (
                        1.0 / delta_time
                    )

            self.last_frame_time = current_time

            # --------------------------------------------
            # Frame ID + FPS tonen (max ~2x/seconde)
            # --------------------------------------------
            # Bij hoge fps kost printen van ELK frame relevante tijd
            # binnen deze callback-thread. Dat vertraagt de callback,
            # waardoor de queue eerder vol raakt. Daarom afgeremd.

            if current_time - self._last_print_time > 0.5:

                print(
                    f"Frame ID: {self.frame_count} | "
                    f"FPS: {self.fps:.1f}",
                    flush=True
                )

                self._last_print_time = current_time

            # --------------------------------------------
            # Naar OpenCV formaat converteren
            # --------------------------------------------

            if (
                frame.get_pixel_format()
                == opencv_display_format
            ):

                display = frame

            else:

                display = (
                    frame.convert_pixel_format(
                        opencv_display_format
                    )
                )

            # --------------------------------------------
            # Omzetten naar OpenCV image
            # --------------------------------------------

            image = display.as_opencv_image()

            # --------------------------------------------
            # In queue plaatsen (NIET blokkerend!)
            # --------------------------------------------
            # Een blokkerende put() hier kan deze callback-thread voor
            # altijd laten vastlopen als de queue vol raakt (bijv. omdat
            # de hoofdloop net gestopt is met lezen na Enter/ESC). Dan
            # kan cam.stop_streaming() nooit meer afronden -> programma
            # loopt vast en de video wordt niet opgeslagen.
            # Oplossing: bij een volle queue het oudste frame weggooien
            # en het nieuwste erin zetten (real-time gedrag).

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

            print(
                f"Onvolledig frame ontvangen "
                f"(status: {status})",
                flush=True
            )

        # ----------------------------------------------------
        # Frame opnieuw klaarzetten voor camera
        # ----------------------------------------------------

        cam.queue_frame(frame)


# ============================================================
# MAIN
# ============================================================

def main():

    print_preamble()

    cam_id = parse_args()

    # --------------------------------------------------------
    # VmbSystem openen
    # --------------------------------------------------------

    with VmbSystem.get_instance():

        # ----------------------------------------------------
        # Camera openen
        # ----------------------------------------------------

        with get_camera(cam_id) as cam:

            print(
                f"Camera geopend: "
                f"{cam.get_name()}"
            )

            # ------------------------------------------------
            # Camera instellen
            # ------------------------------------------------

            setup_camera(cam)

            setup_roi(cam)

            setup_pixel_format(cam)

            # ------------------------------------------------
            # Handler maken
            # ------------------------------------------------

            handler = Handler()

            # ------------------------------------------------
            # Video writer maken
            # ------------------------------------------------

            writer, filename = (
                create_video_writer(cam)
            )

            print(
                f"Opname wordt weggeschreven naar: "
                f"{filename}"
            )

            print()

            # ------------------------------------------------
            # Streaming starten
            # ------------------------------------------------

            try:

                cam.start_streaming(
                    handler=handler,
                    buffer_count=10
                )

                print(
                    "Streaming gestart, "
                    "wachten op frames..."
                )

                print(
                    "Druk op Enter of ESC om te stoppen."
                )

                print()

                # --------------------------------------------
                # OpenCV window
                # --------------------------------------------

                window_name = "Mako camera"

                while True:

                    # ----------------------------------------
                    # Frame ophalen
                    # ----------------------------------------

                    try:

                        display = handler.get_image(
                            timeout=0.5
                        )

                    except Empty:

                        # Geen frame beschikbaar
                        key = cv2.waitKey(1) & 0xFF

                        if key == 13 or key == 27:
                            break

                        continue

                    # ----------------------------------------
                    # Frame tonen
                    # ----------------------------------------

                    cv2.imshow(
                        window_name,
                        display
                    )

                    # ----------------------------------------
                    # Frame opslaan
                    # ----------------------------------------

                    writer.write(
                        display
                    )

                    # ----------------------------------------
                    # Toetsen controleren
                    # ----------------------------------------

                    key = (
                        cv2.waitKey(1)
                        & 0xFF
                    )

                    # Enter of ESC
                    if key == 13 or key == 27:

                        break

                    # ----------------------------------------
                    # Controleren of window gesloten is
                    # ----------------------------------------

                    try:

                        if (
                            cv2.getWindowProperty(
                                window_name,
                                cv2.WND_PROP_VISIBLE
                            )
                            < 1
                        ):

                            break

                    except cv2.error:

                        break

            finally:

                # ------------------------------------------------
                # Streaming stoppen
                # ------------------------------------------------

                cam.stop_streaming()

                # ------------------------------------------------
                # Video opslaan
                # ------------------------------------------------

                writer.release()

                # ------------------------------------------------
                # OpenCV sluiten
                # ------------------------------------------------

                cv2.destroyAllWindows()

                print()

                print(
                    "Video-opname afgesloten "
                    "en opgeslagen."
                )

                print(
                    f"Totaal ontvangen frames: "
                    f"{handler.frame_count}"
                )


# ============================================================
# PROGRAMMA STARTEN
# ============================================================

if __name__ == '__main__':
    main()