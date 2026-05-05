#!/bin/bash
DATASETS=("Deepfakes" "Face2Face" "FaceSwap" "NeuralTextures" "FaceShifter")
OUTPUT="C:/deepfake-detector/data/ffpp"
MAX=20

for d in "${DATASETS[@]}"; do
    attempt=0
    while [ $attempt -lt $MAX ]; do
        attempt=$((attempt + 1))
        echo ""
        echo "=== [$d] Attempt $attempt ==="
        echo "" | python download.py "$OUTPUT" -d "$d" -c c40 -t videos --server EU2
        code=$?
        if [ $code -eq 0 ]; then
            echo "[$d] Complete."
            break
        fi
        echo "[$d] Failed (exit $code). Retrying in 15s..."
        sleep 15
    done
    if [ $attempt -ge $MAX ]; then
        echo "[$d] Gave up after $MAX attempts."
    fi
done

echo ""
echo "=== All done ==="
echo "original:      $(ls $OUTPUT/original_sequences/youtube/c40/videos/*.mp4 2>/dev/null | wc -l)/1000"
echo "Deepfakes:     $(ls $OUTPUT/manipulated_sequences/Deepfakes/c40/videos/*.mp4 2>/dev/null | wc -l)/1000"
echo "Face2Face:     $(ls $OUTPUT/manipulated_sequences/Face2Face/c40/videos/*.mp4 2>/dev/null | wc -l)/1000"
echo "FaceSwap:      $(ls $OUTPUT/manipulated_sequences/FaceSwap/c40/videos/*.mp4 2>/dev/null | wc -l)/1000"
echo "NeuralTextures:$(ls $OUTPUT/manipulated_sequences/NeuralTextures/c40/videos/*.mp4 2>/dev/null | wc -l)/1000"
echo "FaceShifter:   $(ls $OUTPUT/manipulated_sequences/FaceShifter/c40/videos/*.mp4 2>/dev/null | wc -l)/1000"
