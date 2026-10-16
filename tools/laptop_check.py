#!/usr/bin/env python3
"""Laptop readiness check before GPU training (Windows, WSL2 or Linux).

    python tools/laptop_check.py

Reports GPU, VRAM, driver, temperature, power limits, disk, RAM and OS,
then says whether training can start and which preset to use. It only
reads information; it changes nothing on the machine.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

OK, WARN, FAIL = "✅", "⚠️ ", "❌"


def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except Exception:
        return None


def detect_os() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if "microsoft" in platform.uname().release.lower():
        return "wsl"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def main() -> int:
    problems = 0
    osname = detect_os()
    print(f"İşletim sistemi: {osname} ({platform.platform()})")
    print(f"Python: {sys.version.split()[0]}")
    if osname == "windows":
        print(f"{WARN} Eğitim (k2/icefall) Windows'ta doğrudan çalışmaz; WSL2 Ubuntu içinde çalıştırın:")
        print("     PowerShell (yönetici):  wsl --install -d Ubuntu-22.04   → yeniden başlatın")
        print("     Sonra Ubuntu terminalinde bu scripti tekrar çalıştırın.")
    if osname == "macos":
        print(f"{FAIL} NVIDIA GPU gerekiyor; macOS'ta CUDA yok.")
        return 1

    smi = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi") else None)
    if not smi:
        print(f"{FAIL} nvidia-smi bulunamadı. NVIDIA sürücüsünü kurun (WSL2 için sürücü Windows tarafına kurulur).")
        return 1
    q = run([smi, "--query-gpu=name,memory.total,driver_version,temperature.gpu,power.draw,power.limit,"
                  "power.max_limit,utilization.gpu,clocks.max.sm", "--format=csv,noheader,nounits"])
    if not q:
        print(f"{FAIL} nvidia-smi çalışmadı. Sürücü/WSL GPU desteğini kontrol edin.")
        return 1
    name, mem, driver, temp, pdraw, plimit, pmax, util, clk = [x.strip() for x in q.splitlines()[0].split(",")]
    print(f"GPU: {name} | VRAM {mem} MiB | sürücü {driver} | sıcaklık {temp} °C | güç {pdraw}/{plimit} W | kullanım %{util}")
    try:
        vram = int(float(mem))
    except ValueError:
        vram = 0
    if vram < 7000:
        print(f"{WARN} VRAM 8 GB'tan az: 'small' preset ve düşük max-duration kullanın.")
    else:
        print(f"{OK} VRAM yeterli: 'small' preset rahat, 'medium' preset max-duration düşürülerek mümkün.")
    try:
        t = float(temp)
        if t >= 70:
            print(f"{WARN} Boştayken {t:.0f} °C yüksek. Fanları/havalandırmayı kontrol edin, sert zemin veya soğutucu altlık kullanın.")
            problems += 1
        else:
            print(f"{OK} Boşta sıcaklık normal ({t:.0f} °C).")
    except ValueError:
        pass
    if pmax not in ("[N/A]", "N/A", "") and plimit not in ("[N/A]", "N/A", ""):
        print(f"ℹ️  Güç limiti ayarlanabilir görünüyor ({plimit} W, en fazla {pmax} W). "
              "Daha serin çalışma için: sudo nvidia-smi -pl <daha düşük W> (laptoplarda çoğu zaman desteklenmez).")

    total, used, free = shutil.disk_usage(os.path.expanduser("~"))
    free_gb = free / 1e9
    flag = OK if free_gb >= 120 else (WARN if free_gb >= 60 else FAIL)
    print(f"{flag} Boş disk: {free_gb:.0f} GB (pilot için ≥60 GB, tam eğitim için ≥150 GB önerilir)")
    if free_gb < 60:
        problems += 1

    try:
        import psutil  # type: ignore

        ram_gb = psutil.virtual_memory().total / 1e9
    except Exception:
        ram_gb = None
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                ram_gb = int(f.readline().split()[1]) / 1e6
    if ram_gb:
        flag = OK if ram_gb >= 15 else WARN
        print(f"{flag} RAM: {ram_gb:.0f} GB" + ("" if ram_gb >= 15 else " (WSL2'de .wslconfig ile belleği artırın)"))

    if osname == "wsl":
        print("ℹ️  WSL2: Windows'ta güç planını 'En iyi performans' yerine 'Dengeli' tutmak ısınmayı azaltır.")
    print("ℹ️  Laptop prizde olmalı. Mümkünse BIOS/üretici uygulamasından pil şarj sınırını (%60–80) açın.")

    print()
    if problems:
        print(f"{WARN} {problems} uyarı var; yukarıdakileri düzeltip yeniden çalıştırın.")
    else:
        print(f"{OK} Eğitime hazır. Sıradaki adım: stt/README.md → pilot eğitimi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
