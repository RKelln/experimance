#!/bin/bash
# Generate repomix outputs for different scopes
# Usage: ./scripts/repomix_profiles.sh [profile]
# Profiles: all, core, common, core, display, audio, agent, image_server, health, transition

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/repomix-outputs"

mkdir -p "$OUTPUT_DIR"

# Common base patterns to always include
COMMON_INCLUDE="libs/common/**"
CORE_PATTERNS="**/schemas.py,**/config.py,**/__init__.py,**/__main__.py"

generate_profile() {
    local name=$1
    local include=$2
    local output="$OUTPUT_DIR/repomix-$name.xml"
    
    echo "📦 Generating $name profile..."
    npx repomix --include "$include" --output "$output" --style xml
    
    # Show stats
    local tokens=$(grep -o 'tokens' "$output" 2>/dev/null | wc -l || echo "?")
    local size=$(du -h "$output" | cut -f1)
    echo "   ✅ $output ($size)"
}

case "${1:-menu}" in
    full)
        echo "📦 Generating full repo (using config file)..."
        npx repomix --output "$OUTPUT_DIR/repomix-full.xml"
        ;;
    all)
        # Generate all profiles
        echo "📦 Generating all profiles..."
        $0 full
        $0 core
        $0 common
        $0 services
        echo ""
        echo "✅ All profiles generated in $OUTPUT_DIR/"
        ls -lh "$OUTPUT_DIR"/*.xml
        ;;
    core)
        generate_profile "core" "$COMMON_INCLUDE,$CORE_PATTERNS"
        ;;
    common)
        generate_profile "common" "libs/common/**"
        ;;
    core-service)
        generate_profile "service-core" "$COMMON_INCLUDE,services/core/**"
        ;;
    display)
        generate_profile "service-display" "$COMMON_INCLUDE,services/display/**"
        ;;
    audio)
        generate_profile "service-audio" "$COMMON_INCLUDE,services/audio/**"
        ;;
    agent)
        generate_profile "service-agent" "$COMMON_INCLUDE,services/agent/**"
        ;;
    image_server|image-server)
        generate_profile "service-image_server" "$COMMON_INCLUDE,services/image_server/**"
        ;;
    health)
        generate_profile "service-health" "$COMMON_INCLUDE,services/health/**"
        ;;
    transition)
        generate_profile "service-transition" "$COMMON_INCLUDE,services/transition/**"
        ;;
    services)
        # Generate all service profiles
        for svc in core display audio agent image_server health transition; do
            generate_profile "service-$svc" "$COMMON_INCLUDE,services/$svc/**"
        done
        ;;
    menu|*)
        echo "Repomix Profile Generator"
        echo "========================="
        echo ""
        echo "Usage: $0 <profile>"
        echo ""
        echo "Profiles:"
        echo "  full          Full repo (~250k tokens)"
        echo "  all           Generate ALL profiles"
        echo "  core          Core architecture only (~30k tokens)"
        echo "  common        Common library only"
        echo "  core-service  Core service"
        echo "  display       Display service + common"
        echo "  audio         Audio service + common"
        echo "  agent         Agent service + common"
        echo "  image_server  Image server + common"
        echo "  health        Health service + common"
        echo "  transition    Transition service + common"
        echo "  services      All individual service profiles"
        echo ""
        echo "Outputs go to: $OUTPUT_DIR/"
        ;;
esac
