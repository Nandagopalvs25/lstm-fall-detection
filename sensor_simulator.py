"""
sensor_simulator.py
====================
Realistic MPU-6050 sensor simulator for testing the fall detection
pipeline WITHOUT physical hardware.

It generates a continuous stream of (ax, ay, az, gx, gy, gz) values
at 200 Hz (same as SisFall), with realistic ADL noise + injected fall
events so you can watch the detector fire.

Usage (standalone):
    python sensor_simulator.py

Or import it:
    from sensor_simulator import SensorSimulator
    sim = SensorSimulator(fs=200)
    for reading in sim.stream():
        ax, ay, az, gx, gy, gz = reading
        ...
"""

import numpy as np
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional
import random


# ── Realistic signal parameters (derived from SisFall paper) ──────────────
@dataclass
class MotionProfile:
    name: str
    accel_mean:  np.ndarray      # (3,) in g
    accel_noise: np.ndarray      # (3,) std
    gyro_mean:   np.ndarray      # (3,) in °/s
    gyro_noise:  np.ndarray      # (3,) std
    duration_s:  float


PROFILES = {
    "standing": MotionProfile(
        "standing",
        accel_mean=np.array([0.0, 0.0, 1.0]),
        accel_noise=np.array([0.02, 0.02, 0.03]),
        gyro_mean=np.zeros(3),
        gyro_noise=np.array([1.0, 1.0, 0.5]),
        duration_s=3.0,
    ),
    "walking": MotionProfile(
        "walking",
        accel_mean=np.array([0.0, 0.0, 1.0]),
        accel_noise=np.array([0.3, 0.15, 0.4]),
        gyro_mean=np.zeros(3),
        gyro_noise=np.array([15.0, 8.0, 5.0]),
        duration_s=4.0,
    ),
    "sitting_down": MotionProfile(
        "sitting_down",
        accel_mean=np.array([0.05, 0.1, 0.95]),
        accel_noise=np.array([0.1, 0.1, 0.15]),
        gyro_mean=np.array([5.0, 0.0, 0.0]),
        gyro_noise=np.array([8.0, 4.0, 3.0]),
        duration_s=2.5,
    ),
}


class FallEvent:
    """Generates a realistic 3-phase fall waveform (prefail → impact → recovery)."""

    @staticmethod
    def generate(fs: int = 200) -> np.ndarray:
        """
        Returns an (N, 6) array representing one complete fall sequence.
        Phase durations are randomised within realistic bounds.
        """
        rng = np.random.default_rng()
        samples = []

        # Phase 1 – Pre-fall stumble (0.3-0.7 s)
        n1 = int(rng.uniform(0.3, 0.7) * fs)
        t1 = np.linspace(0, 1, n1)
        ax = 0.2 * np.sin(2 * np.pi * 2 * t1) + rng.normal(0, 0.05, n1)
        ay = 0.3 * np.sin(2 * np.pi * 1.5 * t1) + rng.normal(0, 0.05, n1)
        az = 1.0 - 0.3 * t1 + rng.normal(0, 0.05, n1)
        gx = 20 * t1 + rng.normal(0, 5, n1)
        gy = 10 * np.sin(2 * np.pi * t1) + rng.normal(0, 3, n1)
        gz = rng.normal(0, 3, n1)
        samples.append(np.column_stack([ax, ay, az, gx, gy, gz]))

        # Phase 2 – Impact (0.1-0.2 s) – HIGH acceleration spike
        n2 = int(rng.uniform(0.1, 0.2) * fs)
        peak = rng.uniform(3.5, 6.0)          # 3.5 – 6 g impact
        envelope = np.exp(-np.linspace(0, 5, n2))
        ax = peak * envelope * rng.choice([-1, 1]) + rng.normal(0, 0.1, n2)
        ay = peak * 0.6 * envelope * rng.choice([-1, 1]) + rng.normal(0, 0.1, n2)
        az = -0.5 + rng.normal(0, 0.2, n2)
        gx = 150 * envelope + rng.normal(0, 10, n2)
        gy = 80 * envelope * rng.choice([-1, 1]) + rng.normal(0, 10, n2)
        gz = 40 * envelope + rng.normal(0, 5, n2)
        samples.append(np.column_stack([ax, ay, az, gx, gy, gz]))

        # Phase 3 – Post-fall lying still (0.5-1.0 s)
        n3 = int(rng.uniform(0.5, 1.0) * fs)
        ax = rng.normal(0.5, 0.05, n3)
        ay = rng.normal(0.8, 0.05, n3)
        az = rng.normal(0.3, 0.05, n3)
        gx = rng.normal(0, 2, n3)
        gy = rng.normal(0, 2, n3)
        gz = rng.normal(0, 1, n3)
        samples.append(np.column_stack([ax, ay, az, gx, gy, gz]))

        return np.vstack(samples)   # shape: (N, 6)


class SensorSimulator:
    """
    Thread-safe, real-time IMU data simulator.

    Parameters
    ----------
    fs              Sampling frequency (Hz). Default 200 to match SisFall.
    fall_interval   Inject a fall every N seconds. Default 30.
    verbose         Print events to console.
    """

    def __init__(self, fs: int = 200, fall_interval: float = 30.0, verbose: bool = True):
        self.fs = fs
        self.fall_interval = fall_interval
        self.verbose = verbose

        self._buffer: deque = deque(maxlen=fs * 5)   # 5 s ring buffer
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._next_fall_at: float = time.time() + fall_interval
        self._fall_queue: deque = deque()             # pre-generated fall samples
        self._event_log: list = []                    # (timestamp, event_name)

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self):
        """Start background data-generation thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()
        if self.verbose:
            print(f"[Simulator] Started at {self.fs} Hz | fall every {self.fall_interval}s")

    def stop(self):
        """Stop the simulator."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.verbose:
            print("[Simulator] Stopped")

    @property
    def is_fall_ground_truth(self) -> bool:
        """True if the samples currently being served are from a fall event."""
        with self._lock:
            return len(self._fall_queue) > 0

    def stream(self, duration_s: Optional[float] = None) -> Iterator[np.ndarray]:
        """
        Yield one (6,) sample at a time, blocking to maintain real-time rate.
        If duration_s is None, streams indefinitely.
        """
        self.start()
        dt = 1.0 / self.fs
        deadline = time.time() + (duration_s or float("inf"))

        while time.time() < deadline:
            start = time.time()

            sample = self._get_next_sample()
            yield sample

            elapsed = time.time() - start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def inject_fall_now(self):
        """Manually trigger an immediate fall event (useful for demos)."""
        fall_data = FallEvent.generate(self.fs)
        with self._lock:
            self._fall_queue.extend(fall_data)
        ts = time.strftime("%H:%M:%S")
        self._event_log.append((ts, "FALL_INJECTED"))
        if self.verbose:
            print(f"[Simulator] ⚡ Fall injected at {ts}")

    def get_events(self) -> list:
        return list(self._event_log)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _generate_loop(self):
        """Background thread: produce ADL samples + schedule falls."""
        profile_names = list(PROFILES.keys())
        current_profile = PROFILES["standing"]
        profile_samples_left = int(current_profile.duration_s * self.fs)

        while self._running:
            # Switch ADL profile when exhausted
            if profile_samples_left <= 0:
                current_profile = PROFILES[random.choice(profile_names)]
                profile_samples_left = int(current_profile.duration_s * self.fs)

            # Schedule automatic fall
            if time.time() >= self._next_fall_at and not self._fall_queue:
                self._next_fall_at = time.time() + self.fall_interval
                fall_data = FallEvent.generate(self.fs)
                with self._lock:
                    self._fall_queue.extend(fall_data)
                ts = time.strftime("%H:%M:%S")
                self._event_log.append((ts, "FALL_AUTO"))
                if self.verbose:
                    print(f"[Simulator] 🔴 Auto fall injected at {ts}")

            # Generate one ADL sample and push to buffer
            p     = current_profile
            accel = p.accel_mean + np.random.normal(0, p.accel_noise)
            gyro  = p.gyro_mean  + np.random.normal(0, p.gyro_noise)
            sample = np.concatenate([accel, gyro]).astype(np.float32)

            with self._lock:
                self._buffer.append(sample)
            profile_samples_left -= 1

            time.sleep(1.0 / self.fs)

    def _get_next_sample(self) -> np.ndarray:
        """Return next sample: fall queue takes priority over ADL buffer."""
        with self._lock:
            if self._fall_queue:
                return np.array(self._fall_queue.popleft(), dtype=np.float32)
            if self._buffer:
                return np.array(self._buffer[-1], dtype=np.float32)

        # Fallback: standing noise
        return np.array(
            [0.0, 0.0, 1.0] + list(np.random.normal(0, 1, 3)),
            dtype=np.float32,
        )


# ── Standalone test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Sensor Simulator – standalone test (10 seconds)")
    print("-" * 50)

    sim = SensorSimulator(fs=200, fall_interval=5.0, verbose=True)

    count = 0
    for sample in sim.stream(duration_s=10.0):
        if count % 200 == 0:   # print once per second
            ax, ay, az, gx, gy, gz = sample
            mag = np.sqrt(ax**2 + ay**2 + az**2)
            print(f"t={count//200:3d}s | "
                  f"accel=({ax:+.2f}, {ay:+.2f}, {az:+.2f}) g | "
                  f"|a|={mag:.2f}g | "
                  f"gyro=({gx:+5.1f}, {gy:+5.1f}, {gz:+5.1f}) °/s")
        count += 1

    print("\nEvents:", sim.get_events())
    sim.stop()
    print("Done.")
