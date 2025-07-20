# Fresh install on new Ubuntu machine

## 1 | First boot in Windows 11 (10 min)

    BIOS / EC update
    *Settings ⇢ Recovery ⇢ Advanced start-up ⇢ UEFI Firmware Settings ⇢ Update.*

    Turn off Fast Startup – Windows’ hybrid-hibernate leaves the Wi-Fi/BT radio half-initialised, so Linux can’t claim it.
    *Control Panel ⇢ Power Options ⇢ Choose what the power buttons do ⇢ untick “Turn on fast startup” ➜ Save.*

    Full Shutdown (⚠ not reboot).
    This releases the Bluetooth/Wi-Fi firmware so Linux sees the devices on next power-up.

## 2 | BIOS tweaks before installing Linux (3 min)

    Press DEL at power-on.
    
    For Ryzen iGPU systems:
    UMA Frame Buffer Size:	*Advanced ⇢ AMD CBS ⇢ NBIO ⇢ UMC*	8 GB	Prevent OOM on 1024² Lightning & multiple ControlNets.
    Fan curve (optional):	*Advanced ⇢ Smart Fan*	“Silent” preset, then lift 80 °C point to 65 % duty	Drops idle to ≈31 dBA without thermal throttling.
    Secure Boot	            *Boot ⇢ Secure Boot	Off* (or enrol your own keys)	ROCm DKMS builds are simpler with SB off.
    
    Save & exit.

## 3 | Install Linux (30 min)

    Boot from USB installer.
    *Install Ubuntu 24.04 LTS* (or later).
    
    Partitioning:
    - No swap partition (use zram instead).
    
    Install third-party software (Wi-Fi, graphics, etc.).
    
    Finish installation and reboot.

    ```bash
    # 1. system up to date
    sudo apt update && sudo apt full-upgrade

    # 2. Vulkan & VA-API accel for video playback
    sudo apt install mesa-vulkan-drivers mesa-va-drivers libvulkan1

    # 3. Create the directtory you want experimance to live
    mkdir -p Documents/art
    cd Documents/art

    # clone the repo
    git clone https://github.com/RKelln/experimance.git

    cd experimance/experimance
    ```

## 4 | Deploy the application

You can deploy in dev (development) or prod (production) mode. See the [infrastructure README](../README.md) for details. In production services are controlled by systemctl, so in general you'll want to install in dev mode, you can install in production mode afterwards.

```bash
./infra/scripts/deploy.sh experimance install dev
```

This will install all dependencies, including `uv` (used for python package management and virtual environments) and `pyenv` (used for managing Python versions) and all Ubuntu packages needed.

