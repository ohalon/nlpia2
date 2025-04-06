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

# msg = "Roger has 5 tennis balls. "
# msg += "He buys 2 more cans of tennis balls. "
# msg += "Each can has 3 tennis balls. \n"
# msg += "How many tennis balls does he have now?"


send_prompt(
    context_prompt="You are an honest virtual assistant (bot)",
    prompt=msg)


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
