# import re
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
import re
import sys

import pandas as pd
from plotly.offline import plot as plot_html
# import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns

from matplotlib import pyplot as plt

sns.set_style('whitegrid')


LLM_PDF = '2303.18223 - A Survey of LLMs.pdf'
FORMFEED = chr(12)
FORMFEED_BYTE = FORMFEED.encode('utf8')


def normalize_name(name, max_len=None):
    name = name.split('(')[0].lower().strip()
    name = name.strip('-').strip('_').strip()
    name = name.split('[')[0].strip()
    return re.sub(r'\W+', '', name)[:max_len]


def extract_tables(pdf_path=LLM_PDF):
    """ FIXME: Only extracts a couple rows/columns """
    from tabula import read_pdf  # doesn't work well
    return read_pdf(pdf_path, pages="all")


def extract_text(pdf_path=LLM_PDF, write_file=True, page_sep=FORMFEED, header='', footer='\n\n' + '_' * 80 + '\n\n'):
    """ UNUSED: Extract plain text from PDF document, repeat `header` and/or `footer` on each page (for semantic search) """
    import fitz  # pip install PyMuPDF
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = header or ''
        text += page.get_text()  # get plain text (is in UTF-8)
        text += footer or ''

        # blocks = page.get_text_blocks()
        pages.append(text)

    return (page_sep or '\n').join(pages)


FOSS_ORG = {
    'T5': ['https://huggingface.co/t5-large', 'Google', ''],
    'mT5': ['https://https://huggingface.co/google/mt5-large', 'Google', ''],
    'PanGu-α': ['https://huggingface.co/sunzeyeah/pangu-13B', 'PCNL', ''],
    'CPM-2': ['https://huggingface.co/mymusise/CPM-GPT2', 'Tsinghua University', ''],
    'T0': ['https://huggingface.co/bigscience/T0', 'Hugging Face', ''],
    'GPT-NeoX-20B': ['https://huggingface.co/EleutherAI/gpt-neox-20b', 'EleutherAI', ''],
    'CodeGen': ['https://huggingface.co/Salesforce/codegen-16B-multi', 'Salesforce', ''],
    'Tk-Instruct': ['https://huggingface.co/allenai/tk-instruct-11b-def', 'AllenAI', ''],
    'UL2': ['https://huggingface.co/google/flan-ul2', 'Google', ''],
    'OPT': ['https://huggingface.co/facebook/opt-66b', 'Facebook', ''],
    'NLLB': ['https://huggingface.co/facebook/nllb-200-3.3B', 'Meta', ''],
    'BLOOM': ['https://huggingface.co/bigscience/bloom', 'Hugging Face', ''],
    'GLM-10b': ['https://huggingface.co/THUDM/glm-10b', 'Tsinghua University', ''],
    'GLM': ['https://huggingface.co/THUDM/glm-large-chinese', 'Tsinghua University', ''],
    'Flan-T5': ['https://huggingface.co/google/flan-t5-xxl', 'Google', ''],
    'mT0': ['https://huggingface.co/bigscience/bloomz', 'Hugging Face', ''],
    # 'Galactica-mini': ['https://huggingface.co/facebook/galactica-125m', 'Meta', ''],
    # 'Galactica-base': ['https://huggingface.co/facebook/galactica-1.3b', 'Meta', ''],
    # 'Galactica-standard': ['https://huggingface.co/facebook/galactica-6.7b', 'Meta', ''],
    # 'Galactica-large': ['https://huggingface.co/facebook/galactica-30b', 'Meta', ''],
    # 'Galactica-huge': ['https://huggingface.co/facebook/galactica-120b', 'Meta', ''],
    'Galactica': ['https://huggingface.co/facebook/galactica-120b', 'Meta', ''],
    'BLOOMZ': ['https://huggingface.co/bigscience/bloomz', 'Hugging Face', ''],
    'OPT-IML': ['https://huggingface.co/HuggingFaceH4/opt-iml-max-30b', 'Hugging Face', ''],
    'Pythia': ['https://github.com/EleutherAI/pythia', 'EleutherAI', ''],
    'LLaMA': ['https://github.com/juncongmoo/pyllama', 'Google', ''],
    'Vicuna': ['https://vicuna.lmsys.org/', 'Berkeley+CMU+Stanford+UCSD', ''],
    'Koala': ['https://vicuna.lmsys.org/', 'Berkeley', ''],
    'GShard': ['', '', ''],
    'GPT-3': ['https://openai.com', 'OpenAI', ''],
    'LaMDA': ['', '', ''],
    'HyperCLOVA': ['', '', ''],
    'Codex': ['', '', ''],
    'ERNIE 3.0': ['', '', ''],
    'Jurassic-1': ['', '', ''],
    'FLAN': ['', '', ''],
    'MT-NLG': ['', '', ''],
    'Yuan 1.0': ['', '', ''],
    'Anthropic': ['', '', ''],
    'WebGPT': ['', '', ''],
    'Gopher': ['', '', ''],
    'ERNIE 3.0 Titan': ['', '', ''],
    'GLaM': ['', '', ''],
    'InstructGPT': ['', 'OpenAI', ''],
    'AlphaCode': ['', '', ''],
    'Chinchilla': ['', '', ''],
    'PaLM': ['', 'Google', ''],
    'Cohere': ['', '', ''],
    'YaLM': ['', '', ''],
    'AlexaTM': ['', '', ''],
    'Luminous': ['', '', ''],
    'Sparrow': ['', '', ''],
    'WeLM': ['', '', ''],
    'U-PaLM': ['', 'Google', ''],
    'Flan-PaLM': ['https://huggingface.co/google/flan-t5-xxl', 'Google', ''],
    'Flan-U-PaLM': ['', 'Google', ''],
    'Alpaca': ['https://github.com/tatsu-lab/stanford_alpaca/', 'Stanford', ''],
    'GPT-4': ['https://openai.com', 'OpenAI', ''],
    'PanGU-Σ': ['', '', ''],
    # Added by HL:
}
DF_FOSS_ORG = pd.DataFrame(FOSS_ORG).T
DF_FOSS_ORG.columns = 'Source Organization Paper'.split()
DF_FOSS_ORG['Open'] = True
DF_FOSS_ORG.index = list(map(normalize_name, DF_FOSS_ORG.index.values))

ADD_LLMS = [
    {
        'Name': 'AutoGPT',
        'Paper': (
            'https://www.researchgate.net/profile/Mohamed-Fezari-2/publication/370107237_'
            'From_GPT_to_AutoGPT_a_Brief_Attention_in_NLP_Processing_using_DL/links/643fd87a2eca706c8b6d151b/'
            'From-GPT-to-AutoGPT-a-Brief-Attention-in-NLP-Processing-using-DL.pdf'),
        'Parents': ['AgentGPT'],
        'Description': (
            'IL8N of AgentGPT. Seems to be compatible with the Chinese language. '
            'Uses GPT-3 to connect to break a task into achievable tasks and query APIs to accomplish those tasks.'),
    }, {
        'Name': 'AgentGPT',
        'Source': 'https://github.com/reworkd/AgentGPT',
        'Organization': 'Reworkd',
        'Description': 'Uses GPT-3 to connect to break a task into achievable tasks and query APIs to accomplish those tasks.',
    }, {
        'Name': 'Claude',
        'Source': 'https://anthropic.org',
        'Organization': 'Anthropic',
        'Open': False,
        'Release': datetime(2023, 6, 1),
        'Description': (
            'Claims to use "Constitutional AI" and "harmless training" to respond appropriately to '
            'malicious or toxic conversation partners'),
    }, {
        'Source': 'https://github.com/NVIDIA/Megatron-LM',
        'Organization': 'NVIDIA',
        'Paper': 'https://arxiv.org/pdf/2205.05198.pdf',
        'Name': 'Megatron-LM',
        'Release': datetime(2022, 5, 1),
        'Size': 8_300_000_000,
        'Open': True,
    }
]

DF_ADD_LLMS = pd.DataFrame(ADD_LLMS)
DF_ADD_LLMS.index = list(map(normalize_name, DF_ADD_LLMS['Name']))


ADD_LLMS_WIKI_2024 = [
    {
        'Name': 'Falcon 180B',
        'Release': '2023-09-01',
        'Organization': 'Technology Innovation Institute',
        'Size': 180000000000,
        'CorpusSize': '3.5 trillion tokens[184]',
        'TrainingCost': False,
        'Openness': 'Proprietary',
        'License': 'Falcon 180B TII license',
        'Open': False,
        'Description': False,
        'Size_str': '180 billion[184]'
    }, {
        'Name': 'Claude 2.1',
        'Release': '2023-11-01',
        'Organization': 'Anthropic',
        'Size': 100_000_000_000,  # Hobson estimate
        'CorpusSize': 'Unknown',
        'TrainingCost': 'Unknown',
        'Openness': 'Proprietary',
        'License': 'Proprietary',
        'Open': False,
        'Description': 'Used in Claude chatbot. Has a context window of 200,000 tokens, or ~500 pages.[186]',
        'Size_str': 'Unknown'
    }, {
        'Name': 'Grok-1',
        'Release': '2023-11-01',
        'Organization': 'x.AI',
        'Size': 0,
        'CorpusSize': 'Unknown',
        'TrainingCost': 'Unknown',
        'Openness': 'Proprietary',
        'License': 'Proprietary',
        'Open': False,
        'Description': 'Used in Grok chatbot. Grok-1 has a context length of 8,192 tokens and has access to X (Twitter).[187]',
        'Size_str': 'Unknown'
    }, {
        'Name': 'Gemini 1.0',
        'Release': '2023-12-01',
        'Organization': 'Google DeepMind',
        'Size': 0,
        'CorpusSize': 'Unknown',
        'TrainingCost': 'Unknown',
        'Openness': 'Proprietary',
        'License': 'Proprietary',
        'Open': False,
        'Description': 'Multimodal model, comes in three sizes. Used in the chatbot of the same name.[188]',
        'Size_str': 'Unknown'
    }, {
        'Name': 'Phi-2',
        'Release': '2023-12-01',
        'Organization': 'Microsoft',
        'Size': 2700000000,
        'CorpusSize': '1.4T tokens',
        'TrainingCost': 'Unknown',
        'License': 'MIT',
        'Open': True,
        'Openness': 'Open Source',
        'Description': (
            'So-called small language model, that "matches or outperforms models up to 25x larger", '
            'trained on "textbook-quality" data based on the paper "Textbooks Are All You Need". '
            'Model training took "14 days on 96 A100 GPUs".[191]'
        ),
        'Size_str': '2.7B'
    }, {
        'Name': 'Mixtral 8x7B',
        'Release': '2023-12-01',
        'Organization': 'Mistral AI',
        'Size': 46700000000,
        'CorpusSize': 'Unknown',
        'TrainingCost': 'Unknown',
        'License': 'Apache 2.0',
        'Open': True,
        'Openness': 'Open Source',
        'Description': (
            'Mixture of experts model, outperforms GPT-3.5 and Llama 2 70B on many benchmarks. '
            'All weights were released via torrent.[190] ',
            '[huggingface](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)'
        ),
        'Size_str': '46.7B total, 12.9B parameters per token[189]'
    }, {
        'Name': 'Eagle 7B',
        'Release': '2024-01-01',
        'Organization': 'RWKV',
        'Size': 7520000000,
        'CorpusSize': '1.1T tokens',
        'TrainingCost': 'Unknown',
        'License': 'Apache 2.0',
        'Openness': 'Open Source',
        'Open': True,
        'Description': 'An "attention-free" linear transformer based on RWKV-v5 architecture.[192]',
        'Size_str': '7.52B'
    }, {
        'Name': 'Gemini 1.5',
        'Release': '2024-02-01',
        'Organization': 'Google DeepMind',
        'Size': 0,
        'CorpusSize': 'Unknown',
        'TrainingCost': 'Unknown',
        'Opennes': 'Proprietary',
        'License': 'Proprietary',
        'Open': False,
        'Description': (
            'Multimodal model, based on a Mixture-of-Experts (MoE) architecture. '
            'Context window increased to 1 million tokens, though only 128k will be available for developers.[193]'
        ),
        'Size_str': 'Unknown'
    }, {
        'Name': 'Gemma',
        'Release': '2024-02-01',
        'Organization': 'Google DeepMind',
        'Size': 7000000000,
        'CorpusSize': '6T tokens',
        'TrainingCost': 'Unknown',
        'License': 'Apache 2.0[194]',
        'Openness': 'Open Source',
        'Open': True,
        'Description': False,
        'Size_str': '2B and 7B'
    }, {
        'Name': 'Claude 3',
        'Release': '2024-03-01',
        'Organization': 'Anthropic',
        'Size': 175000000000,
        'CorpusSize': 'Unknown',
        'TrainingCost': 'Unknown',
        'Openness': 'Proprietary',
        'License': 'Proprietary',
        'Open': False,
        'Description': 'Includes three models, Haiku, Sonnet, and Opus.[195]',
        'Size_str': 'approx 175B (https://arxiv.org/abs/2302.07459)'}]

DF_ADD_LLMS_2024 = pd.DataFrame(ADD_LLMS_WIKI_2024)
DF_ADD_LLMS_2024.index = list(map(normalize_name, DF_ADD_LLMS_2024['Name']))


def size_str2ints(size_str):
    """ Convert '7.25B' -> 7_250_000_000'

    >>> size_str2int('1.23T and 7e8')
    [1230000000000, 700000000]
    """
    nums = re.findall(r'(\d+[.]?\d*[eE]?\d*)?\s*([Tt]|[Tt]rillion|[Bb]|[Bb]illion|M|[Mm]illion|[Kk]|[Tt]ousand)?', size_str)
    ints = []
    for num, tens in nums:
        if num:
            tens = dict(
                t=1e12, b=1e9, m=1e6, k=1e3,
                trillion=1e12, billion=1e9, million=1e6, thousand=1e3,
            ).get(tens.strip().lower(), 1.)
            ints.append(int(float(num or 0) * (tens or 0)))
    return ints


def size_str2int(size_str):
    """ Convert '7.25B and 3T' -> 3_000_000_000_000'

    >>> size_str2int('1.23T and 7e8')
    1230000000000
    """
    return max(size_str2ints(size_str) or [0])


def scrape_wikipedia(
        url='https://en.wikipedia.org/wiki/Large_language_model',
        match=2,  # r'large\s*language\s*model',
        columns='Name Release Organization Size CorpusSize TrainingCost Openness Description'.split()):
    if isinstance(match, int):
        dfs = pd.read_html(url)[match:]
    else:
        dfs = pd.read_html(url, match=match)
    df = dfs[0]
    df.columns = columns + list(df.columns)[len(columns):]
    max_len = max(map(len, DF_FOSS_ORG.index.values))
    df.index = list(map(normalize_name, df['Name'], [max_len] * len(df)))
    df = df.fillna(False)
    dfdict = df.to_dict()
    dfdict['Size'].update(dict(claude3='approx 175B (https://arxiv.org/abs/2302.07459)'))
    dfdict['Openness'].update(dict(mixtral8x7b='https://huggingface.co/mistralai/Mixtral-8x7B-v0.1'))
    df = pd.DataFrame(dfdict)
    df['Release'] = pd.to_datetime(df['Release'].apply(strip_footnotes)).dt.date()
    df['Size_str'] = df['Size'].copy()
    df['Size'] = df['Size'].apply(strip_footnotes)
    df['Size'] = df['Size'].apply(lambda s: size_str2int(s) if s and size_str2int(s) else 0)
    df = df.sort_values(['Release', 'Size'])
    return df


def strip_footnotes(s):
    """ remove '[123]' substrings from s

    >>> strip_footnotes('October[203] 2021[141] ')
    'October 2021'
    """
    if not isinstance(s, str):
        return s
    return re.sub(r'\[\d+\]', '', s).strip()


def scrape_llm_survey(readme='https://github.com/rucaibox/llmsurvey'):
    """ Scatterplot of LLM size vs release date """
    df = pd.read_html(
        readme, match='Release Time',
        parse_dates=True)[0].dropna()
    df.columns = 'Open,Name,Release,Size,Reference'.split(',')
    df = df.sort_values('Release')

    links = pd.read_html(
        readme, match='Release Time',
        extract_links='body')[0].dropna()
    links = links['Link']['Link'].values
    df['Link'] = list(zip(*links))[1]

    # Typo corrections, cleaning, estimation of missing values
    df['OpenOrig'] = df['Open'].copy()
    df['Proprietary'] = df['Open'].str.lower().str.startswith('p').copy()
    df['Open'] = ~df['Proprietary']
    if 'Openness' not in df.columns:
        df['Openness'] = ['Open Source' if isopen else 'Proprietary' for isopen in df['Open']]
    df['Name'] = df['Name'].replace({'Galatica': 'Galactica'})
    df.set_index('Name', inplace=True, drop=False)
    df['Size'] = df['Size'].replace({r'^[^0-9\s]+$': str(8 * int(df['Size']['GPT-3']))}, regex=True)
    df['Size'] = df['Size'].astype(int) * 1_000_000_000
    return df


def merge_llm_dfs(dfs=None):
    """ Combine open source model information with closed source info from LLM Survey paper """
    if dfs is None:
        dfs = [
            scrape_llm_survey(),
            DF_FOSS_ORG.copy(),  # has different columns
            scrape_wikipedia(),  # duplicated in DF_ADD_LLMS_2024
            DF_ADD_LLMS,
            # DF_ADD_LLMS_2024,
        ]
    for i, df in enumerate(dfs):
        dfs[i].index = list(map(normalize_name, df.index.values))
    df = pd.concat(dfs, axis=1)
    return df


def plot_llm_sizes(
        df='https://github.com/rucaibox/llmsurvey',
        x='Release', y='Size', symbol='Openness', size=12,
        template="seaborn",
        text='index',
        dest='llm_sizes_scatter.html', display=True,
        font_size=18,
        width=2000, height=500, log_x=False, log_y=True,
        **kwargs):
    """ Scatterplot of LLM size vs release date """
    if isinstance(df, (str, Path, TextIOWrapper)):
        df = scrape_llm_survey(readme=df)
    df = df.drop_duplicates(subset=['Release', 'Size'], keep='last')
    if 'Openness' not in df.columns:
        df['Openness'] = [
            'Open Source' if isopen else 'Proprietary'
            for isopen in df['Open']]
    if isinstance(size, (int, float)):
        size = [size] * len(df)
    if isinstance(text, str):
        text = getattr(df, text, text)
    df['Size (trainable parameters)'] = df['Size']
    if dest and dest.lower().endswith('html'):
        fig = px.scatter(
            df,
            x='Release', y='Size (trainable parameters)',
            symbol=symbol, size=size,
            log_x=log_x, log_y=log_y,
            width=width, height=height,
            text=text,
            **kwargs)
        fig.update_layout(
            font_family="Gravitas One,Arial Bold,Open Sans,Arial",
            font_size=font_size,
            legend=dict(
                title=None,
                yanchor="top", y=0.98,
                xanchor="left", x=0.02)
        )
        fig.update_traces(textposition='middle center')
        # scatter = go.Scatter(x=x, y=y, line=None, fill=color)
        plot_html(
            fig, show_link=False, validate=True, output_type='file',
            filename=dest,
            image=None, image_width=width, image_height=height,
            include_plotlyjs=True, include_mathjax=False,
            config=None,
            # autoplay=True, animation_opts=None
        )
    else:
        # df.plot(ax=ax, kind='scatter', x='Release', legend=True, y='Size',
        #         style='Openness',
        #         palette={'Open Source': 'teal', 'Proprietary': 'red'},
        #         markers={'Open Source': 'o', 'Proprietary': 's'},
        #         sizes={'Open Source': 12, 'Proprietary': 10},
        #         rot=-65)
        fig = plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=df,
            x='Release', legend=True, y='Size',
            style='Openness',
            hue='Openness',
        )
        # palette={'Open Source': 'teal', 'Proprietary': 'red'},
        # markers={'Open Source': 'o', 'Proprietary': 's'},
        # sizes={'Open Source': 12, 'Proprietary': 10}
        fig = plt.gcf()
        if display:
            plt.show()

    return fig


if __name__ == '__main__':
    args = list(sys.argv[1:])
    if not len(args):
        args += ['llm_sizes_scatter.html']

    dest = '_'.join(args)
    df = scrape_llm_survey()
    df = df.sort_values(['Release', 'Size'])
    df = df.drop_duplicates(['Release', 'Size'], keep='last')
    df = df.sort_values(['Release'])
    IGNORE_NAMES = [
        'T5', 'mT5',
        'ERNIE 3.0 Titan', 'WebGPT', 'BLOOMZ', 'LaMDA',
        'Cohere', 'CodeGeeX', 'CodeGenX', 'Jurassic-1', 'Koala']
    for name in IGNORE_NAMES:
        if name in df.index:
            df = df.drop(axis=1, index=name)

    # process dates from github repo table strings
    df['Release'] = df['Release'].str.split('/')
    df['Release'] = df['Release'].apply(lambda x: datetime(int(x[0]), int(x[1]), 1))
    df['Release'] = df['Release'].dt.date.astype(str)
    df = df.sort_values('Release')

    # append some new rows gleaned from Wikipedia and manually curated by Hobs
    df = pd.concat([df, DF_ADD_LLMS, DF_ADD_LLMS_2024])

    # cleanup the dates from Wikipedia (NaTs and )
    df['Release'] = df['Release'].apply(strip_footnotes)
    df['Release'] = pd.to_datetime(df['Release']).dt.date.astype(str)

    df = df.sort_values(['Open', 'Release'])  # so that open models will render first with the blue cicles instead of red diamonds
    isnat = df['Release'].str.startswith('N')
    df = df.drop(index=df.index.values[isnat])
    df['DisplayName'] = df['Name'].replace({'OPT': 'OPT  .', 'BLOOM': '.  BLOOM'})
    df = df.set_index('DisplayName')
    fig = plot_llm_sizes(
        df, opacity=.2,
        x='Release', y='Size', color='Open', symbol='Openness',
        dest=dest, display=True,
        width=1400, height=700, log_x=False, log_y=True)

"""  Example Plotly Plots
df.columns
df.columns[2] = 'Release'

plt.grid('on')
plt.show()

pip install plotly
from plotly.offline.offline import _plot_html
import plotly
plotly.offline.offline.plot?
plotly.offline.offline.plot?
from plotly.offline.offline import plot as _plot_html
from plotly.graph_objs import Scatter, Layout
from plotly.graph_objs.scatter import Marker
from plotly.graph_objs.layout import XAxis, YAxis
from nlpia2.constants import SRC_DATA_PATH

np = pd.np

PLOTLY_HTML = \"\"\"
<html>
  <head>
    <meta charset="utf-8" />
    <!-- <meta http-equiv="Content-Type" content="text/html; charset=utf-8"> -->
    <script type="text/javascript">
    {plotlyjs}
    </script>
  </head>
  <body>
    {plotlyhtml}
  </body>
</html>
\"\"\"

DEFAULT_PLOTLY_CONFIG = {
    'staticPlot': False,  # no interactivity, for export or image generation
    'workspace': False,  # we're in the workspace, so need toolbar etc
    'editable': False,  # we can edit titles, move annotations, etc
    'autosizable': False,  # plot will respect layout.autosize=true and infer its container size
    'fillFrame': False,  # if we DO autosize, do we fill the container or the screen?
    'scrollZoom': False,  # mousewheel or two-finger scroll zooms the plot
    'doubleClick': 'reset+autosize',  # double click interaction (false, 'reset', 'autosize' or 'reset+autosize')
    'showTips': True,  # new users see some hints about interactivity
    'showLink': True,  # link to open this plot in plotly
    'sendData': True,  # if we show a link, does it contain data or just link to a plotly file?
    'linkText': 'Edit chart',  # text appearing in the sendData link
    'displayModeBar': 'true',  # display the modebar (true, false, or 'hover')
    'displaylogo': False,  # add the plotly logo on the end of the modebar
    'plot3dPixelRatio': 2,  # increase the pixel ratio for 3D plot images
    'setBackground': 'opaque'  # fn to add the background color to a different container or 'opaque'
                               # to ensure there's white behind it
}
import os

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style?
sns.set_style('white grid')
sns.set_style('whitegrid')
from nlpia2 import constants
constants.SRC_DATA_DIR
pwd
import plotly.graph_objs as go

plot_html([go.Scatter(x=[1, 2, 3], y=[3, 2, 6])], filename='my-graph.html')
import plot.graph_objs as go

more my-graph.html
"""
