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


## 3 | Install Linux and the project (30 min)

    Boot from USB installer.
    *Install Ubuntu 24.04 LTS* (or later).
    
    Partitioning:
    - No swap partition (use zram instead).
    
    Install third-party software (Wi-Fi, graphics, etc.).
    
    Finish installation and reboot.

    ```bash
    # 1. system up to date
    sudo apt update && sudo apt full-upgrade

    # 2. Install git
    sudo apt install git

    # 3. Set git credentials
    git config --global user.name "Your Name"
    git config --global user.email "your.email@example.com"

    # 4. Create the directtory you want experimance to live
    mkdir -p Documents/art
    cd Documents/art

    # 5. clone the repo
    git clone https://github.com/RKelln/experimance.git

    cd experimance/experimance

    # create symlink for easy access
    cd ~
    ln -s Documents/art/experimance/experimance/
    cd experimance
    ```


## 4 | Deploy the application

You can deploy in dev (development) or prod (production) mode. See the [infrastructure README](../README.md) for details. In production services are controlled by systemctl, so in general you'll want to install in dev mode, you can install in production mode afterwards.

```bash
./infra/scripts/deploy.sh experimance install dev
```

This will install all dependencies, including `uv` (used for python package management and virtual environments) and `pyenv` (used for managing Python versions) and all Ubuntu packages needed.


### 4.1 | .env file

Create or copy a .env file for your project and put into `projects/<project_name>/`.


## 5 | SSH remote access (key only)

1. **Install OpenSSH Server on the Target Machine**

    On the machine you want to access remotely (the "target"), run:

    ```bash
    sudo apt update
    sudo apt install openssh-server
    sudo systemctl enable --now ssh
    ```

    To check the SSH service status:
    ```bash
    sudo systemctl status ssh
    ```

2. **Find the Target Machine’s Local IP Address**

    On the target machine, run:
    ```bash
    hostname -I
    ```
    Note the IP address (e.g., `192.168.1.42`).

3. **Test SSH Access from the Source Machine**

    On your existing Ubuntu machine (the "source"), run:
    ```bash
    ssh <username>@<target-ip>
    # Example:
    ssh experimance@192.168.1.42
    ```
    Accept the fingerprint prompt and enter the password when asked.

4. **(Optional) Set Up Passwordless SSH Login**

    On the source machine:
    ```bash
    ssh-keygen   # Press Enter to accept defaults
    ssh-copy-id <username>@<target-ip>
    # Example:
    ssh-copy-id experimance@192.168.1.42
    ```
    Now you can SSH without a password.

**Troubleshooting:**
- Ensure both machines are on the same network and can ping each other.
- If SSH fails, check firewall settings:
  ```bash
  sudo ufw allow ssh
  sudo ufw status
  ```
- If you change the default SSH port, specify it with `-p <port>`.


## 6 | Creating a New 'experimance' User (with sudo access)

1. **Check if the user already exists:**

    ```bash
    id experimance
    ```
    If you see "no such user," continue below.

2. **Create the user and set a password:**

    ```bash
    sudo adduser experimance
    ```
    Follow the prompts to set a password and (optionally) user info.

3. **Add the user to the sudo group:**

    ```bash
    sudo usermod -aG sudo experimance
    ```

4. **(Optional) Switch to the new user:**

    ```bash
    su - experimance
    ```

5. **(Optional) Add home dirs:**
   When logged in as eexperimance user if home directory empty:
   ```bash
   xdg-user-dirs-update
   ```


## 7 | Lock down SSH

    In the project root:
    ```bash
    sudo infra/scripts/secure_ssh.sh status
    sudo infra/scripts/secure_ssh.sh test-keys
    sudo infra/scripts/secure_ssh.sh secure
    ```


## 8 | Run deploy

    In the project root:

    Production deploy of experimance project:
    ```bash
    sudo infra/scripts/deploy.sh experimance install prod
    ```

    Dev install of fire project:
    ```bash
    infra/scripts/deploy.sh fire install dev
    ```

