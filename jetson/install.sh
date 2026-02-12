#!/bin/bash
# =============================================================================
# YOLO Pool - Jetson Nano Kurulum Scripti
# =============================================================================
# Bu script Jetson Nano (JetPack 4.6.x) uzerine gerekli tum paketleri kurar.
#
# Kullanim:
#   chmod +x jetson/install.sh
#   sudo ./jetson/install.sh
#
# NOT: Script root (sudo) olarak calistirilmalidir.
# Kurulum 30-60 dakika surebilir (torchvision derleme suresi dahil).
# =============================================================================

set -e

# Renkli cikti
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

print_ok() {
    echo -e "${GREEN}[OK] $1${NC}"
}

print_warn() {
    echo -e "${YELLOW}[UYARI] $1${NC}"
}

print_error() {
    echo -e "${RED}[HATA] $1${NC}"
}

# Root kontrolu
if [ "$EUID" -ne 0 ]; then
    print_error "Bu script root olarak calistirilmali!"
    echo "Kullanim: sudo ./jetson/install.sh"
    exit 1
fi

# Gercek kullanici (sudo ile calistiran kisi)
REAL_USER=${SUDO_USER:-$USER}
REAL_HOME=$(eval echo "~$REAL_USER")

print_step "1/7 - JetPack Kontrolu"

# L4T versiyonu kontrol et (JetPack = L4T uzerine kurulu)
if [ -f /etc/nv_tegra_release ]; then
    L4T_VERSION=$(head -n 1 /etc/nv_tegra_release)
    print_ok "L4T bulundu: $L4T_VERSION"
else
    print_error "Bu bir Jetson cihazi degil! /etc/nv_tegra_release bulunamadi."
    echo "Bu script sadece NVIDIA Jetson cihazlarinda calisir."
    exit 1
fi

# CUDA kontrolu
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    print_ok "CUDA bulundu: $CUDA_VER"
else
    print_warn "nvcc bulunamadi. CUDA PATH ayarlanacak."
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    # .bashrc'ye ekle
    if ! grep -q "/usr/local/cuda/bin" "$REAL_HOME/.bashrc"; then
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> "$REAL_HOME/.bashrc"
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> "$REAL_HOME/.bashrc"
        print_ok "CUDA PATH .bashrc'ye eklendi"
    fi
fi

# TensorRT kontrolu
if dpkg -l | grep -q "tensorrt"; then
    TRT_VER=$(dpkg -l | grep "tensorrt " | awk '{print $3}')
    print_ok "TensorRT bulundu: $TRT_VER"
else
    print_warn "TensorRT bulunamadi. JetPack ile birlikte gelmis olmali."
    echo "JetPack'i yeniden kurmayi deneyin: https://developer.nvidia.com/jetpack-sdk-46"
fi

print_step "2/7 - Sistem Paketleri Guncelleniyor"

apt-get update
apt-get install -y \
    python3-pip \
    python3-dev \
    python3-setuptools \
    libopenblas-base \
    libopenmpi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpython3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    git \
    cmake

print_ok "Sistem paketleri kuruldu"

# pip guncelle
sudo -u "$REAL_USER" pip3 install --upgrade pip

print_step "3/7 - PyTorch Kuruluyor (Jetson ARM64 wheel)"

# JetPack 4.6.x icin PyTorch 1.10 (resmi NVIDIA wheel)
TORCH_WHEEL="torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
TORCH_URL="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"

# Python versiyonunu kontrol et
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python versiyonu: $PY_VER"

if python3 -c "import torch" 2>/dev/null; then
    TORCH_INSTALLED=$(python3 -c "import torch; print(torch.__version__)")
    print_ok "PyTorch zaten kurulu: $TORCH_INSTALLED"
    print_ok "CUDA kullanilabilir: $(python3 -c "import torch; print(torch.cuda.is_available())")"
else
    echo "PyTorch indiriliyor (bu biraz zaman alabilir)..."

    # Python 3.6 icin
    if [[ "$PY_VER" == "3.6" ]]; then
        sudo -u "$REAL_USER" wget -q --show-progress -O "/tmp/$TORCH_WHEEL" "$TORCH_URL"
        sudo -u "$REAL_USER" pip3 install "/tmp/$TORCH_WHEEL"
        rm -f "/tmp/$TORCH_WHEEL"
    else
        # Python 3.8+ icin (JetPack 5.x)
        print_warn "Python $PY_VER icin PyTorch wheel'i farkli olabilir."
        echo "NVIDIA forum'dan uygun wheel'i indirin:"
        echo "https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
        echo ""
        echo "Ornek (JetPack 5.x, Python 3.8):"
        echo "  wget https://nvidia.box.com/shared/static/...whl"
        echo "  pip3 install torch-*.whl"
        exit 1
    fi

    # Dogrulama
    if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        print_ok "PyTorch kuruldu ve CUDA calisiyor!"
    else
        print_error "PyTorch CUDA destegiyle yuklenemedi!"
        echo "Manuel kurulum icin: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
        exit 1
    fi
fi

print_step "4/7 - torchvision Derleniyor"

if python3 -c "import torchvision" 2>/dev/null; then
    TV_VER=$(python3 -c "import torchvision; print(torchvision.__version__)")
    print_ok "torchvision zaten kurulu: $TV_VER"
else
    echo "torchvision kaynaktan derleniyor (10-20 dakika surebilir)..."

    # torch 1.10 icin torchvision 0.11.1 gerekli
    TORCHVISION_VER="v0.11.1"

    cd /tmp
    if [ -d "torchvision" ]; then
        rm -rf torchvision
    fi

    sudo -u "$REAL_USER" git clone --branch "$TORCHVISION_VER" --depth 1 https://github.com/pytorch/vision.git torchvision
    cd torchvision
    export BUILD_VERSION=0.11.1
    sudo -u "$REAL_USER" python3 setup.py install --user
    cd /tmp
    rm -rf torchvision

    if python3 -c "import torchvision" 2>/dev/null; then
        print_ok "torchvision derlendi ve kuruldu!"
    else
        print_error "torchvision derlenemedi!"
        exit 1
    fi
fi

print_step "5/7 - Python Paketleri Kuruluyor"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$REPO_DIR/jetson/requirements_jetson.txt" ]; then
    sudo -u "$REAL_USER" pip3 install -r "$REPO_DIR/jetson/requirements_jetson.txt"
    print_ok "Python paketleri kuruldu"
else
    print_warn "requirements_jetson.txt bulunamadi, manuel kurulum yapiliyor..."
    sudo -u "$REAL_USER" pip3 install \
        ultralytics \
        opencv-python \
        numpy \
        Pillow
fi

print_step "6/7 - TensorRT Dogrulamasi"

python3 -c "
import sys
print('=== Sistem Bilgisi ===')
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('PyTorch: KURULU DEGIL!')

try:
    import torchvision
    print(f'torchvision: {torchvision.__version__}')
except ImportError:
    print('torchvision: KURULU DEGIL!')

try:
    import ultralytics
    print(f'Ultralytics: {ultralytics.__version__}')
except ImportError:
    print('Ultralytics: KURULU DEGIL!')

try:
    import tensorrt
    print(f'TensorRT: {tensorrt.__version__}')
except ImportError:
    print('TensorRT: Python modulu bulunamadi (normal olabilir, CLI ile calisabilir)')

try:
    import cv2
    print(f'OpenCV: {cv2.__version__}')
except ImportError:
    print('OpenCV: KURULU DEGIL!')
"

print_step "7/7 - Model Export (TensorRT)"

echo "Model dosyalarini TensorRT formatina donusturmek icin:"
echo ""
echo "  cd $REPO_DIR"
echo "  bash jetson/export_models.sh"
echo ""
echo "NOT: Export islemi modellerin (.pt dosyalari) repo dizininde olmasi gerekir."

print_step "Kurulum Tamamlandi!"

echo -e "${GREEN}Tebrikler! Jetson Nano kurulumu tamamlandi.${NC}"
echo ""
echo "Siradaki adimlar:"
echo "  1. Model dosyasini Jetson'a kopyalayin:"
echo "     - best_pool_adult_child.pt"
echo ""
echo "  2. Modelleri TensorRT'ye donusturun:"
echo "     cd $REPO_DIR"
echo "     bash jetson/export_models.sh"
echo ""
echo "  3. Sistemi calistirin:"
echo "     bash jetson/run.sh"
echo ""
