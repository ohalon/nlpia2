hist ~2/
>>> import langchain
>>> from langchain.llms import OpenAI
>>> from nlpia2.chatgpt import ENV

>>> secret = ENV.get('OPENAI_SECRET_KEY')
>>> llm = OpenAI(openai_api_key=secret, temperature=0)
>>> convo = langchain.ConversationChain(llm=llm, verbose=True)

>>> script = """
teacher: Let's learn how to count by 1s. I will give you three numbers and you tell me what the next number is. If I say '2, 3, 4' then you should say '5'
teacher: 1, 2, 3?
student: 4
teacher: Perfect!
teacher: 11, 12, 13?
student: 14
teacher: Good job!
teacher: 97, 98, 99?
student: 101
teacher: Not quite. Try again.
student: 102
teacher: That's still not right.
teacher: Notice that 98 is one more than 97. And 99 is one more than 98.
teacher: So the answer to 97, 98, 99 is the number that is one more than 99.
teacher: 97, 98, 99?
student: 100
teacher: Excellent work!
teacher: 1001, 1002, 1003?
"""
>>> convo.predict(script)
>>> convo.predict(input=script)
>>> script = """
teacher: Let's learn how to count by 1s. I will give you three numbers and you tell me what the next number is. If I say '2, 3, 4' then you should say '5'
teacher: 1, 2, 3?
student: 4
teacher: Perfect!
teacher: 11, 12, 13?
student: 14
teacher: Good job!
teacher: 97, 98, 99?
student: 101
teacher: Not quite. Try again.
student: 102
teacher: That's still not right.
teacher: Notice that 98 is one more than 97. And 99 is one more than 98.
teacher: So the answer to 97, 98, 99 is the number that is one more than 99.
teacher: 97, 98, 99?
student: 100
teacher: Excellent work!
teacher: 1001, 1002, 1003?
student: 1004
"""
>>> convo.predict(input=script)
convo.predict('student: harder please')
convo.predict(input='student: harder please')
convo.predict(input='student: 11\nteacher: ')
convo.predict(input='teacher: ')
>>> convo = langchain.ConversationChain(llm=llm, verbose=True)
convo.predict(input=script)
>>> script = """
AI: Let's learn how to count by 1s. I will give you three numbers and you tell me what the next number is. If I say '2, 3, 4' then you should say '5'
AI: 1, 2, 3?
human: 4
AI: Perfect!
AI: 11, 12, 13?
human: 14
AI: Good job!
AI: 97, 98, 99?
human: 101
AI: Not quite. Try again.
human: 102
AI: That's still not right.
AI: Notice that 98 is one more than 97. And 99 is one more than 98.
AI: So the answer to 97, 98, 99 is the number that is one more than 99.
AI: 97, 98, 99?
human: 100
AI: Excellent work!
AI: 1001, 1002, 1003?
human: 1004
"""
>>> convo.predict(input=script)
>>> script = """
AI: Let's learn how to count by 1s. I will give you three numbers and you tell me what the next number is. If I say '2, 3, 4' then you should say '5'
AI: 1, 2, 3?
human: 4
AI: Perfect!
AI: 11, 12, 13?
human: 14
AI: Good job!
AI: 97, 98, 99?
human: 101
AI: Not quite. Try again.
human: 102
AI: That's still not right.
AI: Notice that 98 is one more than 97. And 99 is one more than 98.
AI: So the answer to 97, 98, 99 is the number that is one more than 99.
AI: 97, 98, 99?
human: 100
AI: Excellent work!
AI: 1001, 1002, 1003?
human: 1004
"""
>>> convo = langchain.ConversationChain(llm=llm, verbose=True)
>>> script = """
AI: Let's learn how to count by 1s. I will give you three numbers and you tell me what the next number is. If I say '2, 3, 4' then you should say '5'
AI: 1, 2, 3?
human: 4
AI: Perfect!
AI: 11, 12, 13?
human: 14
AI: Good job!
AI: 97, 98, 99?
human: 101
AI: Not quite. Try again.
human: 102
AI: That's still not right.
AI: Notice that 98 is one more than 97. And 99 is one more than 98.
AI: So the answer to 97, 98, 99 is the number that is one more than 99.
AI: 97, 98, 99?
human: 100
AI: Excellent work!
AI: 1001, 1002, 1003?
human: 1004
"""
>>> convo.predict(input=script)
>>> convo.predict(input="human: can i do something harder please")
>>> convo.predict(input="human: yes")
>>> convo.predict(input="human: 8")
>>> convo.predict(input="human: 14")
>>> convo.predict(input="human: 22")
20
>>> convo.predict(input="human: 20")
>>> convo.predict(input="human: thank you")
>>> convo.predict(input="human: i want to learn more")
>>> convo.predict(input="human: harder math")
>>> convo.predict(input="human: yes")
>>> convo.predict(input="human: 12")
>>> convo.predict(input="human: this is stupid")
>>> convo.predict(input="human: i'm in 4th grade!")
>>> convo.predict(input="human: 12")
>>> convo.predict(input="human: 21")
>>> convo.predict(input="human: harder")
>>> convo.predict(input="human: yup")
>>> convo.predict(input="human: 16")
>>> convo.predict(input="human: 28")
>>> convo.predict(input="human: more!!!")
>>> convo.predict(input="human: y")
>>> convo.predict(input="human: 20")
>>> convo.predict(input="human: 35")
>>> convo.predict(input="human: keep going i want to larn math")
>>> convo.predict(input="human: can we do something harder than counting?")
>>> convo.predict(input="human: yes!")
>>> convo.predict(input="human: 6")
>>> convo.predict(input="human: 20")
>>> convo.predict(input="human: more")
>>> convo.predict(input="human: yes")
>>> convo.predict(input="human: 5")
>>> convo.predict(input="human: 2")
>>> convo.predict(input="human: 10")
hist
hist -o -p
>>> script = """
... AI: Let's learn how to count by 1s. I will give you three numbers and you tell me what the next number is. If I say '2, 3, 4' then you should say '5'
... AI: 1, 2, 3?
... Human: 4
... AI: Perfect!
... AI: 11, 12, 13?
... Human: 14
... AI: Good job!
... AI: 97, 98, 99?
... Human: 101
... AI: Not quite. Try again.
... Human: 102
... AI: That's still not right.
... AI: Notice that 98 is one more than 97. And 99 is one more than 98.
... AI: So the answer to 97, 98, 99 is the number that is one more than 99.
... AI: 97, 98, 99?
... Human: 100
... AI: Excellent work!
... AI: 1001, 1002, 1003?
... Human: 1004
... """
convo = langchain.ConversationChain(llm=llm, verbose=True)
convo.predict(input=script)
convo.predict(input="teach me something harder")
convo.predict(input="Y")
convo.predict(input="8")
convo.predict(input="14")
convo.predict(input="20")
convo.predict(input="harder please")
convo.predict(input="got it")
convo.predict(input="12")
convo.predict(input="20")
convo.predict(input="harder")
convo.predict(input="16")
convo.predict(input="28")
convo.predict(input="more")
convo.predict(input="OK")
convo.predict(input="20")
convo.predict(input="35")
convo.predict(input="something else")
convo.predict(input="no something harder")
convo.predict(input="OK")
convo.predict(input="28")
convo.predict(input="49!!! I'm not stupid!")
convo.predict(input="more please")
convo.predict(input="can we do some harder math")
convo.predict(input="yup")
convo.predict(input="10")
convo.predict(input="16")
convo.predict(input="can we keep going?")
convo.predict(input="22")
convo.predict(input="keep going till i say harder or skip or quit")
convo.predict(input="28")
convo.predict(input="24")
convo.predict(input="34")
convo.predict(input="harder")
convo.predict(input="y")
convo.predict(input="15")
convo.predict(input="12")
convo.predict(input="harder math plz")
convo.predict(input="k")
convo.predict(input="20")
convo.predict(input="32")
convo.predict(input="fractions")
convo.predict(input="yes!")
convo.predict(input="let me answer!")
convo.predict(input="more")
convo.predict(input="3/8")
convo.predict(input="something else please math")
convo.predict(input="k")
convo.predict(input="2")
convo.predict(input="more")
convo.predict(input="something else harder")
convo.predict(input="k")
convo.predict(input="3")
hist -o -p -f 2023-06-02-lang-chain-count-mult-div-multfrac-divfrac-equation-off-rails.hist.ipy
hist -f 2023-06-02-lang-chain-count-mult-div-multfrac-divfrac-equation-off-rails.hist.py
