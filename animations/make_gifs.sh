#!/usr/bin/env bash
# Full-resolution GIFs of the four animations for the README (1600x900, 20 fps, two-pass palette).
#
#     python anim_sloshing.py && python real_figure.py && python helix_figure.py && ./make_gifs.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../media
for name in sloshing_problem sloshing_before_after acene_before_after ala10_before_after; do
  ffmpeg -v error -y -i "out/$name.mp4" -vf \
    "split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
    "../media/$name.gif"
  echo "media/$name.gif $(du -h "../media/$name.gif" | cut -f1)"
done
