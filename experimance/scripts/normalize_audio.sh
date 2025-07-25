#!/bin/bash

# Audio Normalization Script for Experimance Project
# Normalizes audio files referenced in layers.json using ffmpeg-normalize

# Note: Removed set -e to continue processing even if individual files fail

# use $ mpv media/audio/environment/normalized/ to play the normalized files

# Handle interrupt signals (Ctrl-C)
trap 'echo -e "\n${RED}[INTERRUPTED]${NC} Script interrupted by user. Exiting..."; exit 130' INT TERM

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LAYERS_JSON="$PROJECT_ROOT/services/audio/config/layers.json"
AUDIO_SOURCE_DIR="$PROJECT_ROOT/media/audio/environment"
AUDIO_OUTPUT_DIR="$PROJECT_ROOT/media/audio/environment/normalized"
BACKUP_LAYERS_JSON="$PROJECT_ROOT/services/audio/config/layers.json.backup"

# Normalization settings (EBU R128 loudness standard)
TARGET_LOUDNESS="-23"  # LUFS (broadcast standard)
TRUE_PEAK="-2.0"       # dBFS (prevent clipping)
LOUDNESS_RANGE="7.0"   # LU (dynamic range)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create output directory
setup_directories() {
    print_status "Setting up directories..."
    
    if [ ! -f "$LAYERS_JSON" ]; then
        print_error "layers.json not found at: $LAYERS_JSON"
        exit 1
    fi
    
    mkdir -p "$AUDIO_OUTPUT_DIR"
    print_success "Output directory created: $AUDIO_OUTPUT_DIR"
}

# Extract audio file paths from layers.json
extract_audio_files() {
    # Extract paths, filter out empty ones, and remove "environment/" prefix
    uvx jq -r '.[] | select(.path != null) | .path' "$LAYERS_JSON" | \
    grep -E '\.(mp3|wav|flac|ogg|m4a)$' | \
    sed 's|^environment/||' | \
    sort | uniq
}

# Normalize a single audio file
normalize_file() {
    local filename="$1"
    local input_file="$AUDIO_SOURCE_DIR/$filename"
    local output_file="$AUDIO_OUTPUT_DIR/$filename"
    local output_dir="$(dirname "$output_file")"
    
    # Create subdirectory if needed
    mkdir -p "$output_dir"
    
    if [ ! -f "$input_file" ]; then
        print_warning "Source file not found: $input_file"
        echo "  Looking for: $input_file"
        #echo "  Directory contents:"
        #ls -la "$(dirname "$input_file")" 2>/dev/null || echo "  Directory does not exist"
        return 1
    fi
    
    if [ -f "$output_file" ]; then
        print_warning "Output file already exists, skipping: $filename"
        return 0
    fi
    
    print_status "Normalizing: $filename"
    
    # Run ffmpeg-normalize with EBU R128 loudness normalization
    if uvx ffmpeg-normalize "$input_file" \
        --loudness-range-target "$LOUDNESS_RANGE" \
        --keep-loudness-range-target \
        --target-level "$TARGET_LOUDNESS" \
        --true-peak "$TRUE_PEAK" \
        --offset 0 \
        --dual-mono \
        --print-stats \
        -c:a libmp3lame \
        -b:a 192k \
        -o "$output_file"; then
        print_success "Normalized: $filename"
        return 0
    else
        print_error "Failed to normalize: $filename"
        return 1
    fi
}

# Update layers.json to use normalized files
update_layers_json() {
    local update_json="$1"
    
    if [ "$update_json" = "true" ]; then
        print_status "Backing up layers.json..."
        cp "$LAYERS_JSON" "$BACKUP_LAYERS_JSON"
        print_success "Backup created: $BACKUP_LAYERS_JSON"
        
        print_status "Updating layers.json to use normalized files..."
        
        # Update paths to point to normalized versions
        uvx jq '
        map(
            if .path != null and (.path | test("\\.(mp3|wav|flac|ogg|m4a)$")) then
                .path = "normalized/" + .path
            else
                .
            end
        )' "$LAYERS_JSON" > "${LAYERS_JSON}.tmp" && mv "${LAYERS_JSON}.tmp" "$LAYERS_JSON"
        
        print_success "Updated layers.json to use normalized audio files"
        print_status "Original backed up to: $BACKUP_LAYERS_JSON"
    fi
}

# Main processing function
process_audio_files() {
    local files_processed=0
    local files_failed=0
    local files_skipped=0
    
    print_status "Starting audio normalization process..."
    print_status "Extracting audio file paths from layers.json..."
    
    local file_list=$(extract_audio_files)
    
    echo "Found files:"
    echo "$file_list"
    echo "---"
    
    while IFS= read -r filename; do
        if [ -n "$filename" ]; then
            echo "Processing file $((files_processed + files_failed + files_skipped + 1)): $filename"
            if normalize_file "$filename"; then
                ((files_processed++))
            else
                if [ -f "$AUDIO_OUTPUT_DIR/$filename" ]; then
                    ((files_skipped++))
                else
                    ((files_failed++))
                fi
            fi
            # Check if we should continue (allows for clean interruption)
            sleep 0.1
        fi
    done <<< "$file_list"
    
    echo
    print_success "Processing complete!"
    echo "  Files processed: $files_processed"
    echo "  Files skipped: $files_skipped"
    echo "  Files failed: $files_failed"
    echo
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -u, --update-json    Update layers.json to use normalized files"
    echo "  -h, --help          Show this help message"
    echo
    echo "Examples:"
    echo "  $0                   # Normalize files only"
    echo "  $0 --update-json     # Normalize files and update layers.json"
    echo
    echo "Settings:"
    echo "  Target Loudness: $TARGET_LOUDNESS LUFS"
    echo "  True Peak: $TRUE_PEAK dBFS"
    echo "  Loudness Range: $LOUDNESS_RANGE LU"
}

# Parse command line arguments
UPDATE_JSON=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--update-json)
            UPDATE_JSON=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "=== Experimance Audio Normalization Script ==="
    echo
    
    setup_directories
    process_audio_files
    update_layers_json "$UPDATE_JSON"
    
    echo
    print_success "All done! Normalized files are in: $AUDIO_OUTPUT_DIR"
    
    if [ "$UPDATE_JSON" = "true" ]; then
        print_status "layers.json has been updated to use normalized files"
        print_status "To revert, restore from: $BACKUP_LAYERS_JSON"
    else
        print_status "To update layers.json to use normalized files, run with --update-json"
    fi
}

# Run main function
main
