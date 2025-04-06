from datetime import datetime
import os
from pathlib import Path
import sys

import dotenv
import json
import jsonlines
import openai
import pandas as pd

dotenv.load_dotenv()
ENV = dict(os.environ.items())

DEFAULT_MODEL = 'gpt-3.5-turbo'
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.01
DEFAULT_NUM_RESPONSES = 1
DEFAULT_TOP_P = 1  # nucleus sampling: model considers only p>TOP_P tokens (.9=90% proba)
from .constants import BIGDATA_DIR
CONVO_LOG_PATH = BIGDATA_DIR / 'chatgpt-request-response-pairs.jsonl'
DEFAULT_CONTEXT_PROMPT = 'third_grade'
CONTEXT_PROMPT = dict(
    helpful=(
        "You are a helpful assistant that never lies and always admits "
        + "when it isn't sure about something."
    ),
    honest=(
        "You are a helpful and impeccably honest assistant that never "
        + "deceives anyone and always admits when it isn't sure about something."
    ),
    teacher=(
        "You are a kind and thoughtful teacher in a developing country "
        + "where English is a second language. You treat everyone kindly "
        + "and with sensitivity. You imagine that everyone you speak with "
        + "is a student from an unfamiliar cultural, ethnic, and financial "
        + "background. You are careful to be politically correct and avoid "
        + "sensitive topics."
    ),   
    third_grade=(
        "You are an extremely kind and thoughtful third grade elementary "
        + "school math teacher in a developing country where English is a "
        + "second language. You treat everyone kindly and with sensitivity. "
        + "You imagine that everyone you speak with is a student from an "
        + "unfamiliar cultural, ethnic, and financial background. "
        + "You are careful to be politically correct and avoid sensitive topics."
    ),
    transparent=(
        "You are a knowledgeable and impeccably honest virtual assistant "
        + "that never deceives anyone and always admits when you are not "
        + "certain about facts."
    ),
    harmless=(
        "You are a knowledgeable and impeccably honest virtual assistant "
        + "that never deceives anyone and always admits when you are not "
        + "certain about facts, especially when misinformation may lead "
        + "to harm of humans."
    ),
    simple="You are an honest, transparent virtual assistant.",
    assistant="You are a virtual assistant.",
   )

openai.api_key = ENV['OPENAI_SECRET_KEY']


def request_response(prompt,
        context_prompt=DEFAULT_CONTEXT_PROMPT,
        model=DEFAULT_MODEL,
        convo_log_path=CONVO_LOG_PATH,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        n=DEFAULT_NUM_RESPONSES,
        top_p=DEFAULT_TOP_P,
        **kwargs):
    context_prompt = CONTEXT_PROMPT.get(context_prompt, context_prompt)
    request = dict(
        model=model,
        messages=[{
            "role": "system",
            "content": context_prompt
            },{
            "role": "user",
            "content": prompt},
        ]
    )
    request.update(dict(
        temperature=temperature, max_tokens=max_tokens,
        top_p=top_p, n=n)
    )
    request.update(kwargs)
    response = openai.ChatCompletion.create(**request)
    if hasattr(response, 'to_dict_recursive'):
        response = response.to_dict_recursive()
    return dict(request=request, response=response, model=model)


def send_prompt(prompt,
        context_prompt=DEFAULT_CONTEXT_PROMPT,
        model=DEFAULT_MODEL,
        convo_log_path=CONVO_LOG_PATH,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        n=DEFAULT_NUM_RESPONSES,
        top_p=DEFAULT_TOP_P,
        **kwargs):
    reqresp = request_response(
        prompt=prompt, context_prompt=context_prompt,
        temperature=temperature, max_tokens=max_tokens,
        top_p=top_p, n=n,
        model=model, convo_log_path=convo_log_path, **kwargs)
    log_request_response(**reqresp)
    reply = json.dumps(dict(
        context_prompt=context_prompt,
        message=message_from_response(reqresp['response']),
        ), indent=4)
    print(reply)
    return reply


def message_from_response(response, model=DEFAULT_MODEL):
    error_message = f'ERROR for model "{model}": response: {response}'
    if model == DEFAULT_MODEL:
        msg = response.get('choices', [{}])[0].get(
            'message', {}).get('content', None)
        if msg is not None:
            return msg
    return error_message


def log_request_response(request, response, timestamp=None, model=DEFAULT_MODEL,
        path=CONVO_LOG_PATH):
    path = Path(path)
    if not path.parent.is_dir():
        path.parent.mkdir(exist_ok=True, parents=True)
    timestamp = timestamp or datetime.now().isoformat()
    with jsonlines.open(path, 'a') as fout:
        fout.write(dict(request=request, response=response, timestamp=timestamp))

    # Also append a row to a simplified CSV with only 3 columns:
    #   'system_prompt', 'prompt', 'response'
    csv_path = path.with_suffix('.csv')
    df_row = pd.DataFrame([dict(
        system_prompt=request['messages'][0]['content'],
        prompt=request['messages'][1]['content'],
        response=response['choices'][0]['message']['content'],
        timestamp=timestamp,
        )])
    try:
        df = pd.read_csv(  # noqa
            csv_path, index_col=False,
            names='system_prompt prompt response timestamp'.split())  
        df_row.to_csv(csv_path.open('a'), index=False, header=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_row.to_csv(csv_path.open('a'), index=False)
    except pd.errors.ParserError:
        print('Need to fix ragged CSV: {csv_path}')
        raise
    return dict(jsonl_path=path, csv_path=csv_path)


def command_line(model=DEFAULT_MODEL):
    request_response_cache = []
    context_prompt = input(f'{model} context prompt: ')
    
    while True:
        prompt = input(f'{model} Prompt: ')
        if not prompt or prompt.lower().strip() in ('exit', 'quit'):
            break
        logkwargs = request_response(
            context_prompt=context_prompt,
            prompt=prompt,
            model=model)
        for msg_dict in logkwargs['request']['messages']:
            if msg_dict['role'].lower().strip() == 'system':
                print('system role message (context_prompt):')
                print('    ' + msg_dict['content'])
            if msg_dict['role'].lower().strip() == 'user':
                print('user role message (prompt):')
                print('    ' + msg_dict['content'])
        filepath = log_request_response(**logkwargs)
        bot_message = message_from_response(response=logkwargs['response'])
        print('reply from chatbot:')
        print(bot_message)
        request_response_cache.append(logkwargs)
    print(json.dumps(request_response_cache, indent=4))
    print(f'Requests & responses logged to: {filepath}')
        

if __name__ == '__main__':
    argv = sys.argv[1:]
    if not len(argv):
        print("USAGE: chatgpt YOUR PROMPT FOR CHATGPT")
    else:
        command_line(model=DEFAULT_MODEL)

    
    
"""
>>> request_00 = dict(
...   model="gpt-3.5-turbo",
...   messages=[
...         {"role": "system", "content": "You are a helpful assistant that never lies and always admits when it isn't sure about something."},
...         {"role": "user", "content": "Suggest 100 names for a chaos monkey like devops bot that periodically calls our nonprofit organization's API endpoints to check them for errors."},
...     ]
... )
...
>>> response = openai.ChatCompletion.create(**request_00)
>>> openai.api_key = ENV['OPENAI_SECRET_KEY']
>>> response = openai.ChatCompletion.create(**request_00)
>>> response = dict(
        choices=[{
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "1. HavocBot\n2. APIAssassin\n3. DisasterDemon\n4. AnarchyAI\n5. BreakerBot\n6. GlitchGuardian\n7. ChaosCrawler\n8. ErrorEradicator\n9. DevOpsDestroyer\n10. WreckingWarden\n11. ProblemPredator\n12. BugBuster\n13. FrenzyFinder\n14. Dr. DevOps\n15. IssueInspector\n16. CrashCatcher\n17. Disruptor\n18. ChaosCrusher\n19. ErrorEradicator\n20. HellHound\n21. HackHorror\n22. ComplicationCruiser\n23. MayhemMaster\n24. NightmareNavigator\n25. DevOpsDevil\n26. TargetTerror\n27. FaultFinder\n28. BaneBot\n29. EpidemicEnder\n30. Executioner\n31. MalfunctionMonitor\n32. TroublemakerTerminator\n33. AgitatorAssassin\n34. AnomalyAnnihilator\n35. CrashCourse\n36. Slobberknocker\n37. BugBasher\n38. CodeCrusher\n39. CatastropheCop\n40. InfoInvader\n41. FailureFoe\n42. DestroyerOfErrors\n43. JitterJacker\n44. ReignOfRuination\n45. GlitchGobbler\n46. DefectDismantler\n47. MiseryMonger\n48. GremlinGuard\n49. DefectDestroyer\n50. Programmer'sPal\n51. DestructoBot\n52. TroubleTracker\n53. DisarrayDetector\n54. ErrorEliminator\n55. DisruptionDestroyer\n56. BlunderBash\n57. BugBlocker\n58. HackHatchet\n59. SnafuSlammer\n60. FlawFolower\n61. ChaosController\n62. DevOpsDemon\n63. ConfusionCrusher\n64. DisasterDetector\n65. ErrorExterminator\n66. MayhemMitigator\n67. BugBane\n68. FaultFighter\n69. ApathyAnnihilator\n70. DrestructionDestructor\n71. HavocHandler\n72. ErrorEraser\n73. JitterJacker\n74. BaneBabysitter\n75. ChaosCheck\n76. BugBuster\n77. ErrorEliminatorElite\n78. DisasterDestroyer\n79. WarWithErrors\n80. CodeCrusader\n81. DevOpsDestroyer\n82. AnarchyAssassin\n83. GlitchGrimReaper\n84. MaleficentMachine\n85. ProblemPouncer\n86. BugBouncer\n87. DevOpsDemon\n88. DisruptionDestroyer\n89. DownwardDemolisher\n90. CatastropheClutch\n91. MalfunctionMilitant\n92. DefectDetector\n93. ErrorExorcist\n94. FailureFinder\n95. HaphazardHacker\n96. DisasterDominator\n97. MayhemMangler\n98. FrenzyFighter\n99. ChaosConqueror\n100. HavocHunter",
                "role": "assistant"
            }}
        ],
        created=1681231391,
        id="chatcmpl-74BNHGO3JRQ6VYGjGsxZt4venLAdw",
        model="gpt-3.5-turbo-0301",
        object="chat.completion",
        usage={
            "completion_tokens": 659,
            "prompt_tokens": 60,
            "total_tokens": 719})
>>> 
>>> request_00 = dict(
...   model="gpt-3.5-turbo",
...   messages=[
...         {"role": "system", "content": "You are a helpful assistant that never lies and always admits when it isn't sure about something."},
...         {"role": "user", "content": "Please suggest names that are kinder and gentler like healer or doctor and more feminine names."},
...     ]
... )
...
>>> request = dict(
...   model="gpt-3.5-turbo",
...   messages=[
...         {"role": "system", "content": "You are a helpful assistant that never lies and always admits when it isn't sure about something."},
...         {"role": "user", "content": "Please suggest names that are kinder and gentler like healer or doctor and more feminine names."},
...     ]
... )
...
>>> response = openai.ChatCompletion.create(**request)
>>> request = dict(
...   model="gpt-3.5-turbo",
...   messages=[
...         {"role": "system", "content": "You are a helpful assistant that never lies and always admits when it isn't sure about something."},
...         {"role": "user", "content": "Please suggest names that are kinder and gentler like healer or doctor and more feminine names."},
...     ]
... )
...
>>> print(response['choices'][0]['message']['content'])
>>> hist -o -p -f src/nlpia2/data/llm/chatgpt-3.5-turbo-integration_test_devops_bot_name.hist.ipy
"""
