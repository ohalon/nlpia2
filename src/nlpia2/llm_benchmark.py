""" Question answer pairs and templates for evaluating LLM intelligence """
import numpy as np
from nlpia2.chatgpt import send_prompt

SYLLABLES = 'eje co fa jek zun be miw rez gir muk ale ci jo '
SYLLABLES += 'se ze pa sib wu ka kef em ebu waj mir ase ze nep '
SYLLABLES += 'iz nat tep ma ka gef je mun man lom sa mo ika '
SYLLABLES += 'he aho be eje le puv le zup cid doj hon be lar zut '
SYLLABLES += 'voj cih ni mid vuk ehe nol ki ke pid ic ma ma di '
SYLLABLES += 'ceg tej nuk am lo no biz lo iza ece jic tur cir ih '
SYLLABLES += 'tak pa kev ad beb ne pav jun ren dal le gep bu ugi '
SYLLABLES += 'ugi ji ruh me ze buw di hi lan ku ge wi fi jir '
SYLLABLES += 'ce uj ti heb ve del co jof ag dek '
SYLLABLES = tuple(set(SYLLABLES.split()))




def random_word(syllables_low=2, syllables_high=4):
    num_syllables = np.random.randint(low=syllables_low, high=syllables_high) 
    word = ''
    for i in range(num_syllables):
        word += np.random.choice(SYLLABLES)
    return word


def random_balls_ints(low=0, high=30):
    num_objects, add_num_containers, num_objects_per_container = np.random.randint(
        low=low, high=high, size=3)
    return dict(num_objects=num_objects,
        add_num_containers=add_num_containers,
        num_objects_per_container=num_objects_per_container,
        answer=num_objects + add_num_containers * num_objects_per_container)

def balls_word_problem(
        agent_first_name='Roger',
        num_objects=5,
        add_num_containers=2,
        num_objects_per_container=3,
        container_name_plural='containers',
        container_name='container',
        object_name_plural='tennis balls',
        afterword="Your answer must be a number NOT a sentence."):
    msg = f"{agent_first_name} has {num_objects} {object_name_plural}. "
    msg += f"They obtain {add_num_containers} more {container_name_plural} of "
    msg += f"{object_name_plural}. Each {container_name} has "
    msg += f"{num_objects_per_container} {object_name_plural}. \n"
    msg += f"How many {object_name_plural} do they have now? \n"
    msg += f"{afterword} \n"
    answer = str(num_objects + num_objects_per_container * add_num_containers)
    return dict(question=msg, answer=answer)


FRACTIONS = {'half': .5, 'one third': .33333333333333, 'one fourth': .25,
    'one quarter': .25, 'one eighth': .125, 'one fifth': .2,
    'one tenth': .1, 'one percent': .01, 'one twentieth': .05}


def juggler_word_problem(
        occupation_name='juggler',
        occupation_verb='juggle',
        fraction_specific='half',
        fraction_adjective='half',
        category_objects='balls',
        specific_objects='golf balls',
        object_adjective='blue',
        num_category_objects=16,
        ):
    msg = f"A {occupation_name} can {occupation_verb} {num_category_objects} "
    msg += f"{category_objects}. {fraction_specific.title()} of the "
    msg += f"{category_objects} are {specific_objects}, "
    msg += f"and {fraction_adjective} are {object_adjective}. "
    msg += f"How many {object_adjective} {specific_objects} are there? "
    answer = str(round(
        FRACTIONS[fraction_specific] * FRACTIONS[fraction_adjective] * num_category_objects))
    return dict(question=msg, answer=answer)


def baseline():
    msg = "Roger has 5 tennis balls. "
    msg += "He buys 2 more cans of tennis balls. "
    msg += "Each can has 3 tennis balls. \n"
    msg += "How many tennis balls does he have now?"

    return send_prompt(
        context_prompt="You are an honest virtual assistant (bot)",
        prompt=msg)


perplexity_ai_examples = [
    dict(
        Q="Suzi has 5 wingeroos. Her girlfriend gives her fiv packets of wingeroos but then takes four back. There are three to a pack. How many wingeroos does she have after all this?",
        A="I'm sorry, I couldn't find any relevant search results for the given question. However, based on the information provided, we can calculate the number of wingeroos Suzi has after the exchange. Suzi starts with 5 wingeroos. Her girlfriend gives her 5 packets of wingeroos, which would be a total of 15 wingeroos (assuming each packet has 3 wingeroos). However, her girlfriend takes 4 packets back, which would be a total of 12 wingeroos. Therefore, after the exchange, Suzi would have 5 + 12 = 17 wingeroos.",
        note="from Wolfram Alpha",
        correct="False",
        accuracy=.5,
    ),
    dict(
        Q="Suzi has 5 pickleballs. She buys 2 more packs of tennis balls. Each pack has 3 in it. How many balls does he have now?",
        A="Suzi has 5 pickleballs and buys 2 packs of tennis balls, each containing 3 balls. Therefore, she has a total of: 5+2⋅3=11",
        note="from Wolfram Alpha, should have asked clarifying questions",
        correct="False",
        accuracy=.95,
    ),
    ]