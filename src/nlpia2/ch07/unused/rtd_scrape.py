import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as convert_to_md

import yaml


ALL_SECTIONS = (
    'Versions', 'Downloads', 'Project Home', 'Tags',
    'Description', 'Repository', 'Project Slug',
    'Last Built', 'Maintainers', 'Badge', 'reStructuredText', 'Markdown', 'HTML', 'Tags', 'Short URLs',
    'Stay Updated', 'Blog', 'Email', 'Subscribe'
    'Resources', 'Tutorial', 'Team', 'Documentation', 'Going Ad-free', 'Site Support', 'Site Status', 'Company',
    'Jobs', 'Advertise with Us', 'Read the Docs for Business', 'Branding & Media Kit', 'Privacy Policy', 'Terms of Service',
)
# SECTIONS = ('Description', 'Repository', 'Project Slug', 'Last Built')


def get_sections(text, markdown=False):  # , sections=SECTIONS):
    """ Given a plain text file find the headings indicated and return the content betwen those headings """
    # pairs = list(zip(ALL_SECTIONS[:-1], ALL_SECTIONS[1:]))
    results = {}
    in_section = None
    for line in text.splitlines():
        line = line.rstrip('\n').rstrip('\r').rstrip('¶')
        stripped = line.lstrip()
        if markdown and stripped and stripped[0] in '#=':
            stripped = stripped.lstrip(stripped[0]).lstrip()
            if stripped:
                in_section = stripped
                results[stripped] = []
        if stripped in ALL_SECTIONS:
            in_section = stripped
        if in_section:
            # still on the first line of the section (header line)
            if in_section not in results:
                in_section = stripped
                results[stripped] = ['']
            elif line or results[in_section][-1]:
                results[in_section].append(line)
        elif line:
            if 'front_matter' not in results:
                results['front_matter'] = [line]
            else:
                results['front_matter'].append(line)
    return results


# from markdownify import markdownify as html2md

# md = html2md(resp.text)


if __name__ == "__main__":
    try:
        pages = yaml.full_load(open('../data/rtd_pages.yml'))
    except IOError:
        pages = yaml.full_load(open('../data/rtd_urls.yml'))

    for i, page in enumerate(pages):
        if not page.get('html_bytes'):
            resp = requests.get(page['url'])
            page['html_bytes'] = resp.content
            page['html'] = resp.text
        soup = BeautifulSoup(page['html_bytes'], parser='lxml')
        text = soup.get_text()
        page['sections'] = get_sections(text)
        page['markdown'] = convert_to_md(page['html'])
        page['sections_markdown'] = get_sections(page['markdown'], markdown=True)
        pages[i] = page
