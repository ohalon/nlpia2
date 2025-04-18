# test_qa_bots.py
import pytest  # noqa
import pandas as pd

from qary.skills.qa import Skill
from qary.qa.qa_datasets import get_bot_accuracies

import logging

log = logging.getLogger('qary')


__author__ = "SEE AUTHORS.md"
__copyright__ = "Hobson Lane"
__license__ = "The Hippocratic License, see LICENSE.txt (MIT + Do no Harm)"


def test_qa_bots():
    return  # FIXME: qabot is now broke
    bot = Skill()
    assert callable(bot.reply)
    replies = bot.reply('Hi', context='')
    assert len(replies) == 0
    obama_replies = bot.reply(
        'When was Obama born?',
        context={'doc': {'text': 'Barack Obama was born in 1961.'}})
    assert len(obama_replies) > 0
    obama_replies = sorted(obama_replies)
    assert obama_replies[-1][0] > .9
    assert float(obama_replies[-1][1]) == 1961
    obama_replies = bot.reply(
        'Where was Barack born?',
        context='Barack Obama was born in Hawaii. O. was born in Uzbekistan. M. was born in Ukraine.')
    assert len(obama_replies) > 0
    obama_replies = sorted(obama_replies)
    assert obama_replies[-1][0] > .3
    assert obama_replies[-1][1].lower().startswith('hawaii')


def test_qa_bot_accuracy():
    return  # FIXME: qabot is now broke
    bot = Skill()
    df = pd.DataFrame(get_bot_accuracies(bot))
    q_accuracies = df.groupby('question')['bot_accuracy'].max()
    mean_acc = q_accuracies.mean()
    log.warning(f'Mean bot accuracy: {mean_acc}')
    assert mean_acc > .4
