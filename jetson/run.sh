#!/bin/bash
# =============================================================================
# YOLO Pool - Jetson Nano Baslatma Scripti
# =============================================================================
# Bu script Jetson Nano uzerinde YOLO Pool sistemini baslatir.
#
# Kullanim:
#   bash jetson/run.sh                    # Varsayilan: gercek zamanli kamera
#   bash jetson/run.sh --video video.mp4  # Video isleme
#   bash jetson/run.sh --conf 0.5         # Farkli guven esigi
#   bash jetson/run.sh --camera 1         # Farkli kamera
#
# Klavye kontrolleri (calisirken):
#   q     = Cikis
#   s     = Ekran goruntusu kaydet
#   +/-   = Guven esigini ayarla
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Varsayilan degerler
MODE="realtime"
VIDEO_INPUT=""
VIDEO_OUTPUT=""
CONF="0.4"
CAMERA="0"

# Argumanlari isle
while [[ $# -gt 0 ]]; do
    case $1 in
        --video)
            MODE="video"
            VIDEO_INPUT="$2"
            shift 2
            ;;
        --output|-o)
            VIDEO_OUTPUT="$2"
            shift 2
            ;;
        --conf)
            CONF="$2"
            shift 2
            ;;
        --camera)
            CAMERA="$2"
            shift 2
            ;;
        --help|-h)
            echo "Kullanim: bash jetson/run.sh [SECENEKLER]"
            echo ""
            echo "Secenekler:"
            echo "  --video DOSYA    Video dosyasi isle (varsayilan: gercek zamanli kamera)"
            echo "  --output DOSYA   Cikis dosyasi (video modu icin)"
            echo "  --conf DEGER     Guven esigi, 0-1 arasi (varsayilan: 0.4)"
            echo "  --camera INDEX   Kamera indeksi (varsayilan: 0)"
            echo "  --help           Bu yardim mesajini goster"
            echo ""
            echo "Ornekler:"
            echo "  bash jetson/run.sh                          # Kamera ile gercek zamanli"
            echo "  bash jetson/run.sh --video havuz.mp4        # Video isle"
            echo "  bash jetson/run.sh --camera 1 --conf 0.5   # 2. kamera, yuksek guven"
            exit 0
            ;;
        *)
            print_error "Bilinmeyen parametre: $1"
            echo "Yardim icin: bash jetson/run.sh --help"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  YOLO Pool - Jetson Nano${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ----------------------------
# 1. Performans modu
# ----------------------------
echo "Performans modu ayarlaniyor..."

if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0 2>/dev/null && print_ok "10W modu (MAXN)" || print_warn "nvpmodel ayarlanamadi"
fi

if command -v jetson_clocks &> /dev/null; then
    sudo jetson_clocks 2>/dev/null && print_ok "Maksimum saat hizi" || print_warn "jetson_clocks ayarlanamadi"
fi

# ----------------------------
# 2. Model kontrolu
# ----------------------------
DETECT_MODEL=""

# TensorRT engine varsa onu kullan, yoksa .pt kullan
if [ -f "best_pool_adult_child.engine" ]; then
    DETECT_MODEL="best_pool_adult_child.engine"
    print_ok "Model: $DETECT_MODEL (TensorRT)"
elif [ -f "best_pool_adult_child.pt" ]; then
    DETECT_MODEL="best_pool_adult_child.pt"
    print_warn "Model: $DETECT_MODEL (PyTorch - yavas!)"
    echo "       Daha hizli calisma icin: bash jetson/export_models.sh"
else
    print_error "Model bulunamadi!"
    echo "Asagidaki dosyalardan birini repo dizinine kopyalayin:"
    echo "  - best_pool_adult_child.engine (onerilen)"
    echo "  - best_pool_adult_child.pt"
    exit 1
fi

# ----------------------------
# 3. Kamera kontrolu (sadece realtime modunda)
# ----------------------------
if [ "$MODE" = "realtime" ]; then
    echo ""
    echo "Kamera kontrol ediliyor..."

    # CSI kamera kontrolu (Raspberry Pi kamera modulu, IMX219, IMX477 vb.)
    CSI_AVAILABLE=false
    if [ -e /dev/video0 ] || ls /dev/video* &>/dev/null; then
        print_ok "Video cihazi bulundu: $(ls /dev/video* 2>/dev/null | tr '\n' ' ')"
    fi

    # nvarguscamerasrc ile CSI kamera testi
    if command -v nvgstcapture-1.0 &> /dev/null; then
        CSI_AVAILABLE=true
        print_ok "CSI kamera destegi mevcut (nvarguscamerasrc)"
    fi

    # USB kamera testi
    if [ -e "/dev/video${CAMERA}" ]; then
        print_ok "Kamera /dev/video${CAMERA} hazir"
    else
        print_warn "Kamera /dev/video${CAMERA} bulunamadi!"
        echo ""
        echo "Bagli kameralar:"
        ls /dev/video* 2>/dev/null || echo "  Hicbir kamera bulunamadi!"
        echo ""
        echo "Cozum:"
        echo "  1. USB kamerayi takin ve tekrar deneyin"
        echo "  2. CSI kamera takili ise: bash jetson/run.sh --camera 0"
        echo "  3. Farkli port deneyin: bash jetson/run.sh --camera 1"
        exit 1
    fi
fi

# ----------------------------
# 4. Calistir
# ----------------------------
echo ""
echo -e "${GREEN}Sistem baslatiliyor...${NC}"
echo ""

if [ "$MODE" = "video" ]; then
    if [ -z "$VIDEO_INPUT" ]; then
        print_error "Video dosyasi belirtilmedi!"
        echo "Kullanim: bash jetson/run.sh --video dosya.mp4"
        exit 1
    fi

    if [ ! -f "$VIDEO_INPUT" ]; then
        print_error "Video dosyasi bulunamadi: $VIDEO_INPUT"
        exit 1
    fi

    # Cikis dosyasi belirtilmemisse otomatik olustur
    if [ -z "$VIDEO_OUTPUT" ]; then
        BASENAME=$(basename "$VIDEO_INPUT" | sed 's/\.[^.]*$//')
        VIDEO_OUTPUT="${BASENAME}_output.mp4"
    fi

    echo "Video isleniyor: $VIDEO_INPUT -> $VIDEO_OUTPUT"
    echo "Model: $DETECT_MODEL | Conf: $CONF"
    echo ""

    python3 detect.py \
        -i "$VIDEO_INPUT" \
        -o "$VIDEO_OUTPUT" \
        --model "$DETECT_MODEL" \
        --conf "$CONF"

    echo ""
    print_ok "Video islendi: $VIDEO_OUTPUT"

else
    echo "Gercek zamanli kamera modu baslatiliyor..."
    echo "Model: $DETECT_MODEL | Conf: $CONF | Kamera: $CAMERA"
    echo ""
    echo "Kontroller: q=Cikis | s=Screenshot | +/-=Guven esigi"
    echo ""

    python3 realtime.py \
        --model "$DETECT_MODEL" \
        --conf "$CONF" \
        --camera "$CAMERA" \
        --width 640 \
        --height 480
fi
