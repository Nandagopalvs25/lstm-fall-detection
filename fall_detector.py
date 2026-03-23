"""
fall_detector.py
================
Core inference engine for the BiLSTM Fall Detection system.
Runs on Raspberry Pi using TensorFlow Lite.

Supports:
  • Simulated sensor data (no hardware needed)
  • Real MPU-6050 via I2C (enable with USE_REAL_SENSOR = True)

Usage:
    python fall_detector.py                    # simulated
    python fall_detector.py --real-sensor      # real MPU-6050
    python fall_detector.py --demo-fall        # inject immediate fall

Architecture:
    Sensor (200 Hz) → Downsample (20 Hz) → Sliding Window (2 s)
    → TFLite BiLSTM → Probability → Threshold → Alert
"""

import argparse
import json
import os
import sys
import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from datetime import datetime


# ── Configuration ─────────────────────────────────────────────────────────
MODEL_PATH      = "./saved_models/fall_detector.tflite"
NORM_PARAMS     = "./saved_models/norm_params.json"
USE_REAL_SENSOR = False          # flip to True when MPU-6050 is connected

ORIGINAL_FS     = 200            # sensor Hz
TARGET_FS       = 20             # model Hz  (matches SisFall training)
DOWNSAMPLE      = ORIGINAL_FS // TARGET_FS
WINDOW_SIZE     = 40             # samples at 20 Hz = 2 seconds
N_FEATURES      = 6              # ax, ay, az, gx, gy, gz
WINDOW_STRIDE   = 10             # 0.5 s stride
THRESHOLD       = 0.35           # overridden by norm_params.json if present


# ── Data classes ──────────────────────────────────────────────────────────
@dataclass
class DetectionResult:
    timestamp:   str
    probability: float
    is_fall:     bool
    window_id:   int


@dataclass
class SystemStats:
    total_windows:  int = 0
    falls_detected: int = 0
    start_time:     float = field(default_factory=time.time)

    @property
    def uptime_s(self) -> float:
        return time.time() - self.start_time

    @property
    def windows_per_second(self) -> float:
        return self.total_windows / max(1.0, self.uptime_s)


# ── TFLite inference wrapper ──────────────────────────────────────────────
class TFLiteModel:
    """Thin wrapper around tflite_runtime / TF-Lite interpreter."""

    def __init__(self, model_path: str):
        self._interpreter = None
        self._keras_model  = None
        self._mode         = None

        # ── Option 1: tflite_runtime (lightweight, Pi-native) ─────────────
        try:
            import tflite_runtime.interpreter as tflite
            interp = tflite.Interpreter(model_path=model_path)
            interp.allocate_tensors()
            self._interpreter = interp
            self._mode        = "tflite_runtime"
            print("✓ Loaded model via tflite_runtime")
        except (ImportError, Exception):
            pass

        # ── Option 2: tensorflow tflite (full TF, includes Flex delegate) ──
        if self._interpreter is None:
            try:
                import tensorflow as tf
                interp = tf.lite.Interpreter(model_path=model_path)
                interp.allocate_tensors()
                self._interpreter = interp
                self._mode        = "tf.lite"
                print("✓ Loaded model via tensorflow lite")
            except (ImportError, Exception):
                pass

        # ── Option 3: Keras direct with JAX backend (no TF needed) ──────
        if self._interpreter is None:
            keras_path = model_path.replace(".tflite", ".keras")
            if not os.path.exists(keras_path):
                keras_path = os.path.join(os.path.dirname(model_path), "bilstm_final.keras")
            if os.path.exists(keras_path):
                # Try backends in order: jax (lightest), torch, tensorflow
                for backend in ("jax", "torch", "tensorflow"):
                    try:
                        os.environ.setdefault("KERAS_BACKEND", backend)
                        import keras
                        model = keras.models.load_model(keras_path)
                        # Warm-up pass to confirm it works
                        import numpy as _np
                        model(
                            _np.zeros((1, WINDOW_SIZE, N_FEATURES), dtype=_np.float32),
                            training=False
                        )
                        self._keras_model = model
                        self._mode        = f"keras/{backend}"
                        print(f"✓ Loaded Keras model ({backend} backend): {keras_path}")
                        break
                    except Exception as e:
                        print(f"  Keras/{backend} failed: {e}")
                        # Reset so next iteration can try a different backend
                        import sys
                        for mod in list(sys.modules.keys()):
                            if mod.startswith("keras"):
                                del sys.modules[mod]

        if self._interpreter is None and self._keras_model is None:
            raise RuntimeError(
                "Could not load model via tflite_runtime, tensorflow, or Keras.\n"
                "Install TensorFlow: pip3 install --break-system-packages tensorflow"
            )

        if self._interpreter is not None:
            self._input_idx  = self._interpreter.get_input_details()[0]["index"]
            self._output_idx = self._interpreter.get_output_details()[0]["index"]
            inp_shape        = self._interpreter.get_input_details()[0]["shape"]
            print(f"  Input shape : {inp_shape}  |  mode: {self._mode}")

    def predict(self, window: np.ndarray) -> float:
        """
        window: (WINDOW_SIZE, 6) float32
        returns: fall probability in [0, 1]
        """
        x = window[np.newaxis, ...].astype(np.float32)   # (1, 40, 6)

        if self._keras_model is not None:
            out = self._keras_model(x, training=False)
            return float(np.array(out)[0][0])  # works for JAX, TF, and PyTorch backends

        self._interpreter.set_tensor(self._input_idx, x)
        self._interpreter.invoke()
        return float(self._interpreter.get_tensor(self._output_idx)[0][0])


class DummyModel:
    """
    Fallback when no .tflite file is found.
    Mimics the real model output based on acceleration magnitude –
    useful for pipeline testing without a trained model.
    """

    def __init__(self):
        print("⚠️  No TFLite model found – using physics-based dummy model")
        print("   Copy fall_detector.tflite to ./saved_models/ for real inference")

    def predict(self, window: np.ndarray) -> float:
        accel    = window[:, :3]          # (40, 3)
        mag      = np.linalg.norm(accel, axis=1)   # (40,)
        peak_g   = float(np.max(mag))
        mean_g   = float(np.mean(mag))

        # Heuristic: SisFall falls typically have peak > 2.5 g
        if peak_g > 4.0:
            base = 0.85
        elif peak_g > 3.0:
            base = 0.70
        elif peak_g > 2.5:
            base = 0.55
        elif peak_g > 1.8:
            base = 0.30
        else:
            base = 0.10

        # Add small noise so the output looks like real probabilities
        noise = np.random.normal(0, 0.03)
        return float(np.clip(base + noise, 0.0, 1.0))


# ── Normalisation ──────────────────────────────────────────────────────────
class Normaliser:
    def __init__(self, params_path: str):
        if os.path.exists(params_path):
            with open(params_path) as f:
                p = json.load(f)
            self.mean = np.array(p["mean"], dtype=np.float32)
            self.std  = np.array(p["std"],  dtype=np.float32)
            self.threshold = float(p.get("threshold", THRESHOLD))
            print(f"✓ Loaded norm params (threshold={self.threshold:.3f})")
        else:
            print(f"⚠️  {params_path} not found – using identity normalisation")
            self.mean = np.zeros(6, dtype=np.float32)
            self.std  = np.ones(6,  dtype=np.float32)
            self.threshold = THRESHOLD

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)


# ── Alert callbacks ───────────────────────────────────────────────────────
class AlertManager:
    """Manages fall alert callbacks and debouncing."""

    def __init__(self, cooldown_s: float = 10.0):
        self._callbacks: List[Callable[[DetectionResult], None]] = []
        self._cooldown   = cooldown_s
        self._last_alert = 0.0

    def register(self, fn: Callable[[DetectionResult], None]):
        self._callbacks.append(fn)

    def trigger(self, result: DetectionResult):
        now = time.time()
        if now - self._last_alert < self._cooldown:
            return                               # debounce
        self._last_alert = now
        for fn in self._callbacks:
            try:
                fn(result)
            except Exception as e:
                print(f"[Alert] Error in callback: {e}")


def console_alert(result: DetectionResult):
    print(
        f"\n{'!'*60}\n"
        f"  🚨 FALL DETECTED at {result.timestamp}\n"
        f"     Probability: {result.probability:.1%}\n"
        f"{'!'*60}\n"
    )


def gpio_alert(result: DetectionResult):
    """
    Optional: Blink LED / activate buzzer on Pi GPIO.
    Uncomment and wire GPIO 17 (LED) and GPIO 27 (buzzer).
    """
    try:
        import RPi.GPIO as GPIO
        LED_PIN    = 17
        BUZZER_PIN = 27
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.setup(BUZZER_PIN, GPIO.OUT)

        for _ in range(5):
            GPIO.output(LED_PIN, GPIO.HIGH)
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(0.2)
            GPIO.output(LED_PIN, GPIO.LOW)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(0.2)

        GPIO.cleanup()
    except ImportError:
        pass    # Not on Pi


# ── Main detector ──────────────────────────────────────────────────────────
class FallDetector:
    """
    Real-time fall detection pipeline.

    Data flow:
        raw samples (200 Hz) → downsample buffer → normalise
        → sliding window (40 × 6) → model → probability → alert
    """

    def __init__(
        self,
        model_path: str   = MODEL_PATH,
        norm_path:  str   = NORM_PARAMS,
        use_gpio:   bool  = False,
        verbose:    bool  = True,
    ):
        self.verbose   = verbose
        self.stats     = SystemStats()
        self._window_id = 0

        # ── Load model ────────────────────────────────────────────────────
        if os.path.exists(model_path):
            try:
                self.model = TFLiteModel(model_path)
            except RuntimeError as e:
                print(f"⚠️  TFLite load failed: {e}")
                print("   Falling back to physics-based dummy model.")
                print("   Run: pip3 install tflite-runtime  to enable real inference.")
                self.model = DummyModel()
        else:
            self.model = DummyModel()

        # ── Normaliser ────────────────────────────────────────────────────
        self.norm = Normaliser(norm_path)

        # ── Sliding window buffers ────────────────────────────────────────
        #   raw_buf  : accumulates 200 Hz samples before downsample
        #   win_buf  : downsampled 20 Hz samples for the model window
        self._raw_buf = deque(maxlen=DOWNSAMPLE * WINDOW_SIZE * 2)
        self._win_buf = deque(maxlen=WINDOW_SIZE * 2)
        self._raw_count = 0
        self._win_count_since_last_inference = 0

        # ── Alerts ────────────────────────────────────────────────────────
        self.alerts = AlertManager(cooldown_s=10.0)
        self.alerts.register(console_alert)
        if use_gpio:
            self.alerts.register(gpio_alert)

        # Recent results for dashboard / logging
        self._results: deque = deque(maxlen=500)
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    def push_sample(self, sample: np.ndarray):
        """
        Feed one raw 200 Hz sample (6,) into the pipeline.
        Call this in your sensor loop.
        """
        with self._lock:
            self._raw_buf.append(sample)
            self._raw_count += 1

            # Downsample: keep every DOWNSAMPLE-th raw sample
            if self._raw_count % DOWNSAMPLE == 0:
                self._win_buf.append(sample)
                self._win_count_since_last_inference += 1

                # Run inference when we have a full window AND stride is met
                if (len(self._win_buf) >= WINDOW_SIZE and
                        self._win_count_since_last_inference >= WINDOW_STRIDE):
                    self._run_inference()
                    self._win_count_since_last_inference = 0

    def get_recent_results(self, n: int = 100) -> List[DetectionResult]:
        with self._lock:
            return list(self._results)[-n:]

    def get_latest_probability(self) -> Optional[float]:
        with self._lock:
            if self._results:
                return self._results[-1].probability
        return None

    # ── Internal ─────────────────────────────────────────────────────────

    def _run_inference(self):
        window_raw  = np.array(list(self._win_buf)[-WINDOW_SIZE:], dtype=np.float32)
        window_norm = self.norm.transform(window_raw)

        prob    = self.model.predict(window_norm)
        is_fall = prob >= self.norm.threshold
        ts      = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        result = DetectionResult(
            timestamp   = ts,
            probability = prob,
            is_fall     = is_fall,
            window_id   = self._window_id,
        )

        self._results.append(result)
        self.stats.total_windows += 1
        self._window_id += 1

        if is_fall:
            self.stats.falls_detected += 1
            self.alerts.trigger(result)
        elif self.verbose and self._window_id % 10 == 0:
            print(f"[{ts}] Window {self._window_id:5d} | "
                  f"prob={prob:.3f} | ADL ✓")

    def run_from_simulator(self, duration_s: Optional[float] = None):
        """Convenience: run with simulated data (no real sensor needed)."""
        from sensor_simulator import SensorSimulator

        fall_interval = 20.0 if duration_s is None else max(5.0, duration_s / 3)
        sim = SensorSimulator(fs=ORIGINAL_FS, fall_interval=fall_interval, verbose=False)

        print(f"\n{'='*60}")
        print(f"  BiLSTM Fall Detector – SIMULATION MODE")
        print(f"  Model threshold : {self.norm.threshold:.3f}")
        print(f"  Fall injected every {fall_interval:.0f}s")
        print(f"  Press Ctrl+C to stop")
        print(f"{'='*60}\n")

        try:
            for sample in sim.stream(duration_s=duration_s):
                self.push_sample(sample)
        except KeyboardInterrupt:
            print("\n[Detector] Stopped by user")
        finally:
            sim.stop()
            self._print_summary()

    def run_from_mpu6050(self):
        """Run with real MPU-6050 sensor on Pi GPIO."""
        try:
            import smbus2
        except ImportError:
            print("❌ smbus2 not installed. Run: pip install smbus2")
            return

        MPU_ADDR   = 0x68
        PWR_MGMT_1 = 0x6B
        ACCEL_OUT  = 0x3B
        GYRO_OUT   = 0x43

        bus = smbus2.SMBus(1)
        bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)   # wake up

        def read_word(reg):
            h = bus.read_byte_data(MPU_ADDR, reg)
            l = bus.read_byte_data(MPU_ADDR, reg + 1)
            val = (h << 8) | l
            return val - 65536 if val >= 0x8000 else val

        print("\n[Detector] Real MPU-6050 sensor mode active")
        dt = 1.0 / ORIGINAL_FS

        try:
            while True:
                t0 = time.time()
                ax = read_word(ACCEL_OUT)     / 16384.0   # ±2g range
                ay = read_word(ACCEL_OUT + 2) / 16384.0
                az = read_word(ACCEL_OUT + 4) / 16384.0
                gx = read_word(GYRO_OUT)      / 131.0     # ±250 °/s
                gy = read_word(GYRO_OUT + 2)  / 131.0
                gz = read_word(GYRO_OUT + 4)  / 131.0

                self.push_sample(np.array([ax, ay, az, gx, gy, gz], dtype=np.float32))

                elapsed = time.time() - t0
                if dt - elapsed > 0:
                    time.sleep(dt - elapsed)

        except KeyboardInterrupt:
            print("\n[Detector] Stopped")
            self._print_summary()

    def _print_summary(self):
        s = self.stats
        print(f"\n{'='*60}")
        print(f"  Session Summary")
        print(f"  Uptime        : {s.uptime_s:.1f}s")
        print(f"  Total windows : {s.total_windows}")
        print(f"  Falls detected: {s.falls_detected}")
        print(f"  Throughput    : {s.windows_per_second:.1f} windows/s")
        print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiLSTM Fall Detector")
    parser.add_argument("--real-sensor", action="store_true",
                        help="Use real MPU-6050 sensor (requires smbus2 + hardware)")
    parser.add_argument("--demo-fall",   action="store_true",
                        help="Inject an immediate fall event 5s after start")
    parser.add_argument("--duration",    type=float, default=None,
                        help="Run for N seconds then exit (default: indefinite)")
    parser.add_argument("--gpio",        action="store_true",
                        help="Activate LED/buzzer on GPIO (Raspberry Pi only)")
    args = parser.parse_args()

    detector = FallDetector(use_gpio=args.gpio)

    if args.real_sensor:
        detector.run_from_mpu6050()
    else:
        if args.demo_fall:
            def _inject():
                time.sleep(5)
                from sensor_simulator import SensorSimulator
                print("\n[Demo] Injecting fall in 5 seconds…")
            threading.Thread(target=_inject, daemon=True).start()

        detector.run_from_simulator(duration_s=args.duration)
