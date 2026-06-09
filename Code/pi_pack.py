"""
download_for_pi.py  —  Run this on your LAPTOP (Windows)
=========================================================
Downloads all packages needed to install TTS (pyttsx3 + espeak-ng)
on the Pi WITHOUT internet access.

Usage:
    python download_for_pi.py          # auto-detects Pi arch (ask prompt)
    python download_for_pi.py arm64    # Pi 4 / Pi 5 running 64-bit OS
    python download_for_pi.py armhf    # Pi 3 / Pi 4 running 32-bit OS

After running, you get a folder called  pi_packages/
Transfer the whole folder to your Pi via USB drive or SCP, then run
the install commands printed at the end.
"""

import os
import sys
import urllib.request
import shutil

# ──────────────────────────────────────────────────────────────────────────────
#  PACKAGE DEFINITIONS  (Debian bookworm — matches Raspberry Pi OS 2023+)
# ──────────────────────────────────────────────────────────────────────────────

BASE = "http://ftp.debian.org/debian/pool/main/e/espeak-ng"

# These version numbers match bookworm (deb12). Update if your Pi runs bullseye.
PACKAGES = {
    "arm64": [
        # espeak-ng binary
        f"{BASE}/espeak-ng_1.51+dfsg-10+deb12u2_arm64.deb",
        # shared library
        f"{BASE}/libespeak-ng1_1.51+dfsg-10+deb12u2_arm64.deb",
        # data files (architecture-independent)
        f"{BASE}/espeak-ng-data_1.51+dfsg-10+deb12u2_all.deb",
        # compatibility shim so pyttsx3 finds "espeak" binary
        f"{BASE}/espeak-ng-espeak_1.51+dfsg-10+deb12u2_arm64.deb",
    ],
    "armhf": [
        f"{BASE}/espeak-ng_1.51+dfsg-10+deb12u2_armhf.deb",
        f"{BASE}/libespeak-ng1_1.51+dfsg-10+deb12u2_armhf.deb",
        f"{BASE}/espeak-ng-data_1.51+dfsg-10+deb12u2_all.deb",
        f"{BASE}/espeak-ng-espeak_1.51+dfsg-10+deb12u2_armhf.deb",
    ],
}

OUT_DIR = "pi_packages"


def progress(count, block_size, total):
    pct = min(100, count * block_size * 100 // total) if total > 0 else 0
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)


def download(url: str, dest: str) -> bool:
    filename = url.split("/")[-1]
    path = os.path.join(dest, filename)
    print(f"\n⬇  {filename}")
    try:
        urllib.request.urlretrieve(url, path, reporthook=progress)
        print(f"\r  ✓  saved → {path}               ")
        return True
    except Exception as e:
        print(f"\r  ✗  FAILED: {e}")
        # Try a mirror
        mirror_url = url.replace("ftp.debian.org/debian", "debian.mirror.liteserver.nl/debian")
        print(f"     Trying mirror …")
        try:
            urllib.request.urlretrieve(mirror_url, path, reporthook=progress)
            print(f"\r  ✓  saved via mirror → {path}  ")
            return True
        except Exception as e2:
            print(f"\r  ✗  Mirror also failed: {e2}")
            return False


def download_pyttsx3(dest: str):
    print("\n📦  Downloading pyttsx3 Python wheels …")
    try:
        result = os.system(
            f'pip download pyttsx3 --dest "{dest}" --quiet'
        )
        if result == 0:
            print("  ✓  pyttsx3 wheels downloaded")
        else:
            print("  ✗  pip download failed — try manually:")
            print(f'     pip download pyttsx3 --dest "{dest}"')
    except Exception as e:
        print(f"  ✗  Error: {e}")


def main():
    # Determine architecture
    if len(sys.argv) > 1 and sys.argv[1] in ("arm64", "armhf"):
        arch = sys.argv[1]
    else:
        print("Which Raspberry Pi OS are you using?")
        print("  1 = 64-bit  (Pi 4 / Pi 5 with 64-bit OS)  →  arm64")
        print("  2 = 32-bit  (Pi 3 / Pi 4 with 32-bit OS)  →  armhf")
        print()
        choice = input("Enter 1 or 2: ").strip()
        arch = "arm64" if choice == "1" else "armhf"

    print(f"\nTarget architecture: {arch}")
    print(f"Output folder:       {OUT_DIR}/\n")

    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # Download .deb files
    print("═" * 52)
    print("  Downloading espeak-ng packages")
    print("═" * 52)
    failed = []
    for url in PACKAGES[arch]:
        ok = download(url, OUT_DIR)
        if not ok:
            failed.append(url.split("/")[-1])

    # Download pyttsx3 wheels
    print()
    print("═" * 52)
    download_pyttsx3(OUT_DIR)

    # Summary
    print()
    print("═" * 52)
    if failed:
        print(f"⚠  {len(failed)} file(s) failed to download:")
        for f in failed:
            print(f"     {f}")
        print()
        print("Manual download links (paste into your browser):")
        for url in PACKAGES[arch]:
            if url.split("/")[-1] in failed:
                print(f"  {url}")
    else:
        print("✓  All files downloaded successfully!")

    print()
    print("═" * 52)
    print("NEXT STEPS")
    print("═" * 52)
    print()
    print("1. Copy the pi_packages/ folder to your Pi.")
    print("   Via USB drive:  copy the folder to a USB, plug into Pi")
    print("   Via SCP (if on same WiFi):")
    print("     scp -r pi_packages/ pi@10.42.0.1:~/")
    print()
    print("2. On the Pi, run these commands:")
    print()
    print("   # Install the .deb packages (order matters)")
    print("   cd ~/pi_packages")
    print("   sudo dpkg -i libespeak-ng1_*.deb")
    print("   sudo dpkg -i espeak-ng-data_*.deb")
    print("   sudo dpkg -i espeak-ng_*.deb")
    print("   sudo dpkg -i espeak-ng-espeak_*.deb")
    print()
    print("   # Install pyttsx3 from local wheels")
    print("   pip3 install --no-index --find-links ~/pi_packages pyttsx3")
    print()
    print("   # Test it")
    print('   python3 -c "import pyttsx3; e=pyttsx3.init(); e.say(\'Hello I am Pippo\'); e.runAndWait()"')
    print()
    print("   You should hear sound through your speaker.")
    print()


if __name__ == "__main__":
    main()