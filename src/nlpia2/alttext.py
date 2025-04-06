from pathlib import Path

from transformers import pipeline

from nlpia2.constants import IMAGE_DIR


if not IMAGE_DIR.is_dir():
    username = Path.home().name
    # IMAGE_DIR = Path.home() / f'code/{username}/nlpia-manuscript/manuscript/images'
    IMAGE_DIR = Path.home() / '.nlpia-data/manuscript/images'


pipe = pipeline("image-to-text", model="Salesforce/blip2-opt-2.7b")


def generate_caption(image, max_new_tokens=40):
    """ Generate caption text for an image file path or URL

    >>> image = '/home/hobs/code/hobs/nlpia-manuscript/manuscript/images/ch02/piano_roll.jpg'
    >>> generate_caption(image)
    """
    return pipe(image, max_new_tokens=max_new_tokens)


def caption_images(pipe=pipe, image_dir=IMAGE_DIR, suffixes='png jpg jpeg tif svg drawio'.split()):  # gif ppt pptx ods
    suffixes = ['.' + s[1:] if s[0] == '.' else s for s in suffixes]
    image_captions = []
    for suf in suffixes:
        glob = '**/*' + suf
        for i, p in enumerate(IMAGE_DIR.glob(glob)):
            print(p)
            p = Path(p)
            d = dict(
                caption='',
                path=str(p),
                filename=p.name,
                dirname=p.parent.name,
                suffix=p.suffix,
                suffixes=''.join(p.suffixes),
                is_file=p.is_file(),
                size=p.stat().st_size)
            if p.is_file():
                try:
                    d['caption'] = list(alttext.generate_caption(str(p))[0].values())[0]
                except Exception as e:
                    d['caption'] = str(e)
    return pd.DataFrame(image_captions)

    import yaml
    image_links = yaml.safe_load(open('image_log.yaml'))
    ls
    OFFICIAL_RELATIVE_MANUSCRIPT_DIR
    BASE_DIR
    OFFICIAL_MANUSCRIPT_DIR = BASE_DIR
    OFFICIAL_RELATIVE_MANUSCRIPT_DIR = 'nlpia-manuscript/manuscript'
    for i in range(4):
        OFFICIAL_MANUSCRIPT_DIR = OFFICIAL_MANUSCRIPT_DIR.parent
        if (OFFICIAL_MANUSCRIPT_DIR / OFFICIAL_RELATIVE_MANUSCRIPT_DIR / 'adoc').is_dir():
            break
    OFFICIAL_MANUSCRIPT_DIR = (OFFICIAL_MANUSCRIPT_DIR / OFFICIAL_RELATIVE_MANUSCRIPT_DIR).resolve()
    OFFICIAL_ADOC_DIR = (OFFICIAL_MANUSCRIPT_DIR / 'adoc').resolve()
    OFFICIAL_MANUSCRIPT_DIR
    hist
    df
    new
    more
    old
    pd.DataFrame(old + more)
    df2 = _
    df2.to_csv(IMAGE_DIR / 'image_log_blip_captions_1more.csv')
