#!/usr/bin/env python3
"""
VisionAssist ESP32 Button Listener
====================================
Receives UDP commands from ESP32 and triggers VisionAssist actions.

Commands:
  "voice"    → Start voice listening (short press)
  "describe" → Describe the scene (double press)
  "read"     → Read text / OCR (long press)

Usage:
  # Standalone test:
  python esp32_listener.py

  # Integrated into your project — see integrate() method below.
"""

import socket
import threading
import time

UDP_PORT = 5005
BUFFER_SIZE = 64


class ESP32ButtonListener:
    """
    Listens for UDP commands from ESP32 button controller.
    Thread-safe, non-blocking.
    """

    def __init__(self, port: int = UDP_PORT, on_voice=None, on_describe=None, on_read=None):
        self.port = port
        self.on_voice = on_voice
        self.on_describe = on_describe
        self.on_read = on_read

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", port))
        self._sock.settimeout(0.5)  # Non-blocking with timeout

        self._running = False
        self._thread = None

        # Get local IP for display
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            self._local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            self._local_ip = "unknown"

        print(f"🎮 ESP32 Button Listener initialized")
        print(f"   Listening on UDP port {port}")
        print(f"   Pi IP address: {self._local_ip}")
        print(f"   → Set this IP in your ESP32 sketch as PI_IP")
        print(f"   Commands: voice | describe | read")

    def start(self):
        """Start listening in background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print(f"🎮 ESP32 listener running (UDP :{self.port})")

    def stop(self):
        """Stop listening."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self._sock.close()
        except Exception:
            pass
        print("🎮 ESP32 listener stopped")

    def _listen_loop(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(BUFFER_SIZE)
                command = data.decode("utf-8").strip().lower()
                print(f"🎮 ESP32 [{addr[0]}]: {command}")

                if command == "voice" and self.on_voice:
                    threading.Thread(target=self.on_voice, daemon=True).start()
                elif command == "describe" and self.on_describe:
                    threading.Thread(target=self.on_describe, daemon=True).start()
                elif command == "read" and self.on_read:
                    threading.Thread(target=self.on_read, daemon=True).start()
                else:
                    print(f"🎮 Unknown command or no handler: {command}")

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"🎮 ESP32 listener error: {e}")
                    time.sleep(0.5)

    def poll(self) -> str | None:
        """
        Non-threaded alternative: call this in your main loop.
        Returns command string or None.
        """
        try:
            self._sock.settimeout(0.01)
            data, addr = self._sock.recvfrom(BUFFER_SIZE)
            command = data.decode("utf-8").strip().lower()
            print(f"🎮 ESP32 [{addr[0]}]: {command}")
            return command
        except socket.timeout:
            return None
        except Exception:
            return None


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  ESP32 Button Listener — Test Mode")
    print("=" * 50)

    def on_voice():
        print("🎤 VOICE triggered! (would start listening)")

    def on_describe():
        print("👁️  DESCRIBE triggered! (would describe scene)")

    def on_read():
        print("📖 READ triggered! (would do OCR)")

    listener = ESP32ButtonListener(
        on_voice=on_voice,
        on_describe=on_describe,
        on_read=on_read,
    )
    listener.start()

    print("\nWaiting for ESP32 button presses... (Ctrl+C to quit)\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        listener.stop()
