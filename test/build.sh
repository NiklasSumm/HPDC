echo "Building BitonicSortGPU..."

rm -rf ./build
rm -rf ./bitonic_sort_cuda.cpython-312-x86_64-linux-gnu.so
rm -rf ./bitonic.o

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Build the CUDA extension
echo "Building CUDA extension..."
python3 setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "You can now run: srun -p exercise-gpu --gres=gpu:1 python3 main.py"
else
    echo "Build failed!"
    exit 1
fi
