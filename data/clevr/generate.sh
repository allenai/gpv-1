GPU=$1
TASK=$2
SUBSET=$3
NUM_IMAGES=$4
BLENDER="/home/tanmayg/Applications/blender-2.78c-linux-glibc219-x86_64/blender"
OUTPUT_DIR="/home/tanmayg/Data/gpv"
SPECS_DIR="/home/tanmayg/Code/gpv/data/clevr/specs"
MIN_OBJECTS=1
MAX_OBJECTS=3

# Automatically specify the dataset directory inside OUTPUT_DIR based on provided arguments
DATASET_DIR="${OUTPUT_DIR}/clevr_min_objects_${MIN_OBJECTS}_max_objects_${MAX_OBJECTS}/${TASK}_task/${SUBSET}"
PROPERTIES_JSON="${SPECS_DIR}/${TASK}_task/properties.json"
SHAPE_COLOR_COMBO_JSON="${SPECS_DIR}/${TASK}_task/shape_color_combos/${SUBSET}.json"
IMAGE_DIR="${DATASET_DIR}/images"
SCENE_DIR="${DATASET_DIR}/scenes"
SCENE_JSON="${DATASET_DIR}/scenes.json"

export CUDA_VISIBLE_DEVICES=$GPU
cd "${PWD}/third_party/clevr-dataset-gen/image_generation"
$BLENDER --background -noaudio --python render_images.py -- \
    --min_objects $MIN_OBJECTS \
    --max_objects $MAX_OBJECTS \
    --num_images $NUM_IMAGES \
    --output_image_dir $IMAGE_DIR \
    --output_scene_dir $SCENE_DIR \
    --output_scene_file $SCENE_JSON \
    --properties_json $PROPERTIES_JSON \
    --shape_color_combos_json $SHAPE_COLOR_COMBO_JSON \
    --use_gpu 1