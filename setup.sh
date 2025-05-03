sudo apt remove --purge 'rocm*' 'hip*' 

sudo rm -rf /opt/rocm

sudo mkdir -p /etc/apt/keyrings

wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg

echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.3 jammy main' |   sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update

sudo apt install rocm-hip-sdk

sudo apt install rocm-dkms

export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export ROCM_PATH=/opt/rocm

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
