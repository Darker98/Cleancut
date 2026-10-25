# Caption font

The caption style references **Montserrat Bold** by family name and ffmpeg is
pointed at this folder via `fontsdir=`, so no system install is required — the
`.ttf` just needs to live here.

## Get the font (one-time)

Download **Montserrat Bold** (SIL Open Font License) and save it here as
`Montserrat-Bold.ttf`:

- Google Fonts: https://fonts.google.com/specimen/Montserrat
  (Download family → unzip → copy `Montserrat-Bold.ttf` here)

Alternatively use **Inter Bold** (https://fonts.google.com/specimen/Inter):
save it as `Montserrat-Bold.ttf` *or* update `font_name`/`font_file` in
`pipeline/config.py` (`CaptionConfig`) to point at `Inter-Bold.ttf` and family
name `Inter`.

Expected final path:

```
assets/fonts/Montserrat-Bold.ttf
```

> The family name embedded in the file must match `CaptionConfig.font_name`
> (default `Montserrat`). If captions render in a fallback font, confirm the
> name with any font inspector and update the config.
