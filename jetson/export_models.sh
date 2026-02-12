#!/bin/bash
# =============================================================================
# YOLO Pool - Model Export (PyTorch -> TensorRT)
# =============================================================================
# Bu script .pt model dosyalarini TensorRT .engine formatina donusturur.
# TensorRT modelleri Jetson Nano uzerinde 3-5x daha hizli calisir.
#
# Kullanim:
#   cd yolo_pool/
#   bash jetson/export_models.sh
#
# NOT: Bu islemi Jetson Nano UZERINDE calistirin (baska bilgisayarda degil!)
#      Export islemi model basina 10-30 dakika surebilir.
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Repo dizinine git
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

echo "Calisma dizini: $(pwd)"

# Jetson kontrolu
if [ ! -f /etc/nv_tegra_release ]; then
    print_error "Bu bir Jetson cihazi degil!"
    echo "TensorRT export islemi Jetson uzerinde yapilmalidir."
    echo "Baska bir bilgisayarda olusturulan .engine dosyalari Jetson'da calismaz."
    exit 1
fi

# Maksimum performans modu
print_step "Performans Modu Ayarlaniyor"

if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0
    print_ok "10W modu aktif (MAXN)"
else
    print_warn "nvpmodel bulunamadi, performans modu ayarlanamadi"
fi

if command -v jetson_clocks &> /dev/null; then
    sudo jetson_clocks
    print_ok "jetson_clocks aktif (maksimum CPU/GPU saat hizi)"
else
    print_warn "jetson_clocks bulunamadi"
fi

# Model export
print_step "Model Export (best_pool_adult_child.pt)"

DETECT_MODEL="best_pool_adult_child.pt"
DETECT_ENGINE="best_pool_adult_child.engine"

if [ -f "$DETECT_ENGINE" ]; then
    print_warn "$DETECT_ENGINE zaten mevcut. Tekrar olusturmak icin once silin:"
    echo "  rm $DETECT_ENGINE"
    echo "  bash jetson/export_models.sh"
else
    if [ -f "$DETECT_MODEL" ]; then
        echo "Export basliyor... (10-30 dakika surebilir, sabirl olun)"
        echo ""
        python3 -c "
from ultralytics import YOLO
model = YOLO('$DETECT_MODEL')
model.export(format='engine', half=True, device=0)
print('Export tamamlandi!')
"
        if [ -f "$DETECT_ENGINE" ]; then
            print_ok "$DETECT_ENGINE olusturuldu! ($(du -h "$DETECT_ENGINE" | cut -f1))"
        else
            print_error "Export basarisiz! $DETECT_ENGINE olusturulamadi."
            exit 1
        fi
    else
        print_error "$DETECT_MODEL bulunamadi!"
        echo "Model dosyasini repo dizinine kopyalayin:"
        echo "  scp bilgisayar:~/yolo_pool/$DETECT_MODEL $(pwd)/"
        exit 1
    fi
fi

# Ozet
print_step "Export Tamamlandi!"

echo "Olusturulan dosyalar:"
echo ""
for f in *.engine; do
    if [ -f "$f" ]; then
        echo "  $f  ($(du -h "$f" | cut -f1))"
    fi
done

echo ""
echo "Sistemi baslatmak icin:"
echo "  bash jetson/run.sh"
echo ""
echo "Veya manuel olarak:"
echo "  python3 realtime.py --model best_pool_adult_child.engine --conf 0.4"
echo ""
