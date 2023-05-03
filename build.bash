MODEL_SIZE=13B

reqPkg=(
    build-essential
)
totalPkg=$(dpkg -l | awk '{print $2}')

for pkg in ${reqPkg[@]};
    do
        installed=false
        for installedPkg in ${totalPkg[@]};
            do
                if [[ $pkg == $installedPkg ]]; then
                    echo "[=] $pkg is installed"
                    installed=$true
                fi
            done
        if [[ $installed = false ]]; then
            echo "$pkg is not installed"
            echo "Installing $pkg..."
            sudo apt install $pkg
            echo "[+] $pkg installed"
        fi
    done

# Check if cog is installed
# https://github.com/replicate/cog#install
if ! command -v cog &> /dev/null
then
    echo "- Cog could not be found. Installing..."
    sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
    sudo chmod +x /usr/local/bin/cog
    echo "- Cog installed"
fi

# If hackathon virtual environment does not exist:
if [ ! -d "hackathon" ]; then
    echo "- Creating virtual environment..."
    python3 -m venv hackathon
    source hackathon/bin/activate
    pip install -r requirements.txt
    echo "- Virtual environment created"
fi

# Check if cog_stanford_alpaca is installed
# https://github.com/replicate/cog_stanford_alpaca
if [ ! -d "app/cog_stanford_alpaca" ]; then
    echo "- Cloning cog_stanford_alpaca..."
    cd app
    git clone https://github.com/replicate/cog_stanford_alpaca
    cd ..
    echo "- cog_stanford_alpaca cloned"
fi

# Move unconverted weights to cog_stanford_alpaca if unconverted-weights does not exist
if [ ! -d "app/cog_stanford_alpaca/weights" ]; then
    if [ ! -d "app/cog_stanford_alpaca/unconverted-weights" ]; then
        echo "- Moving unconverted weights..."
        # Check if unconverted weights is a .tar file and if so, extract and then move
        if [ -f "unconverted-weights.tar" ]; then
            tar -xvf unconverted-weights.tar
            echo "- Unconverted weights extracted"
        fi
        mv unconverted-weights app/cog_stanford_alpaca
        echo "- Unconverted weights moved"
    fi
    # Build the weights
    cd app/cog_stanford_alpaca
    sudo cog run python -m transformers.models.llama.convert_llama_weights_to_hf \
    --input_dir unconverted-weights \
    --model_size $MODEL_SIZE \
    --output_dir weights
fi

# python utils/build.py