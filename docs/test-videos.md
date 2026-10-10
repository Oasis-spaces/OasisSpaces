# Test videos

The videos the phone's perception and the pipeline are tested on are published as assets of the GitHub release [test-videos](https://github.com/Oasis-spaces/OasisSpaces/releases/tag/test-videos), not in the repository tree: `videos/` is gitignored so that checkouts (Colab, the app builds) stay small. Download any of them with

```bash
curl -L -o pexels-11299294.mp4 https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-11299294.mp4
```

## Phone recordings

| Asset | What it is |
|---|---|
| [IMG_4138.MOV](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/IMG_4138.MOV) | The pan: a 21 s iPhone video of a bedroom, the source of `spaces/pan-full` |
| [IMG_4182-walkthrough-1080p.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/IMG_4182-walkthrough-1080p.mp4) | The walkthrough: a 93 s iPhone video of a bedroom, the source of `spaces/walkthrough-full`, as a 1080p copy (the 912 MB 4K original is on the Mac in `videos/`) |

The 18 September 2026 capture made with Oasis Capture (`spaces/phone-20260918-0101-0f304b93`) kept only its frames and poses, not a video file.

## Public rooms

Eighteen free-licence room clips from Pexels (Pexels License: free to use, no attribution required), found with the Firecrawl connector on 6 October 2026. `videos/public/SOURCES.md` on the Mac is the same list.

| Asset | Pexels page |
|---|---|
| [pexels-35023423.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-35023423.mp4) | https://www.pexels.com/video/elegant-modern-bedroom-interior-design-35023423/ |
| [pexels-39936361.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-39936361.mp4) | https://www.pexels.com/video/modern-minimalist-bedroom-interior-design-39936361/ |
| [pexels-32144167.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-32144167.mp4) | https://www.pexels.com/video/elegant-vintage-bedroom-with-antique-furniture-32144167/ |
| [pexels-38413578.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-38413578.mp4) | https://www.pexels.com/video/modern-minimalist-bedroom-with-natural-light-38413578/ |
| [pexels-29681899.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-29681899.mp4) | https://www.pexels.com/video/elegant-hotel-suite-with-classic-decor-and-furniture-29681899/ |
| [pexels-34208854.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-34208854.mp4) | https://www.pexels.com/video/cozy-minimalist-bedroom-with-modern-decor-34208854/ |
| [pexels-35308496.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-35308496.mp4) | https://www.pexels.com/video/modern-minimalist-apartment-interior-tour-35308496/ |
| [pexels-7749088.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-7749088.mp4) | https://www.pexels.com/video/tracking-shot-of-a-bedroom-7749088/ |
| [pexels-3769951.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-3769951.mp4) | https://www.pexels.com/video/modern-living-room-3769951/ |
| [pexels-27975847.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-27975847.mp4) | https://www.pexels.com/video/the-kitchen-and-dining-area-of-a-small-apartment-27975847/ |
| [pexels-27975843.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-27975843.mp4) | https://www.pexels.com/video/a-small-apartment-with-a-kitchen-and-dining-area-27975843/ |
| [pexels-37674124.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-37674124.mp4) | https://www.pexels.com/video/modern-luxury-apartment-interior-view-37674124/ |
| [pexels-7578546.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-7578546.mp4) | https://www.pexels.com/video/video-of-a-house-interior-7578546/ |
| [pexels-11299294.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-11299294.mp4) | https://www.pexels.com/video/interior-of-apartment-in-residential-building-11299294/ |
| [pexels-8580864.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-8580864.mp4) | https://www.pexels.com/video/video-of-an-inside-the-house-8580864/ |
| [pexels-7614541.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-7614541.mp4) | https://www.pexels.com/video/a-living-room-at-home-7614541/ |
| [pexels-10135086.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-10135086.mp4) | https://www.pexels.com/video/modern-minimalist-home-interior-10135086/ |
| [pexels-29814959.mp4](https://github.com/Oasis-spaces/OasisSpaces/releases/download/test-videos/pexels-29814959.mp4) | https://www.pexels.com/video/cozy-bedroom-interior-with-modern-decor-29814959/ |

## How they are used

`phonesim --video <clip> --dump <folder> --fps 3` runs the phone's perception over a clip; `tools/phone_video_preview.py` turns a set of such runs into preview videos, contact sheets and a reel (see the Oasis Capture section of the README). The processed rooms and the capture replay are judged with `phonesim spaces/<name>`.
