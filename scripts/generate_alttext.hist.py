from nlpia2 import alttext
from nlpia2.constants import *

IMAGE_DIR = Path('/home/hobs/code/hobs/nlpia-manuscript/manuscript/images')

alts = []
for glob in ['**/*.png', '**/*.jpg']:
    for i, p in enumerate(IMAGE_DIR.glob(glob)):
        if p.is_file():
            print(p)
            try:
                alts.append(list(alttext.generate_caption(str(p))[0].values())[0])
                print('SUCCESS!!!!', alts[-1])
            except Exception as e:
                print(e)
                alts.append(str(p))
for i, p in enumerate(df.path.values):
    if isinstance(alts[i], dict):
        continue

    # binary_fc = open(p, 'rb').read()  # fc aka file_content
    # base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
    # outpath = p.with_suffix('.base64' + p.suffix)
    # outpath.open('wb').write(base64_utf8_str)
    # dataurl = f'data:image/{p.suffix.strip(".")};base64,{base64_utf8_str}'
    for let in 'abcdefg':
        if p.parent.name.startswith('app') and p.parent.name.endswith(f'_{let}'):
            p = p.parent.parent / f'appendix-{let}' / p.name
    print(i, p)
    try:
        alts[i] = alttext.generate_caption(str(p))[0]
    except Exception as e:
        alts[i] = e
    print(alts[i])
len(alts)
len(df)
alts[121:]
pd.DataFrame(alts)
pd.DataFrame(alts[:121])
pd.DataFrame([(str(a) if isinstance(a, Exception) else list(a.values())[0]) for a in alts]).to_csv('alts.csv')
dfalt = pd.DataFrame([(str(a) if isinstance(a, Exception) else list(a.values())[0]) for a in alts])
dfalt
il = Path('../../hobs/nlpia-manuscript/image_log/')
il.is_dir()
il.list()
il.lstat
il.lstat()
dfalt.to_csv(il / 'image_log_alttext.csv')
