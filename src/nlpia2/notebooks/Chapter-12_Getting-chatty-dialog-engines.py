#!/usr/bin/env python
# coding: utf-8

# #### [`Chapter-12_Getting-chatty-dialog-engines`](/home/hobs/code/hobs/nlpia-manuscript/manuscript/adoc/Chapter-12_Getting-chatty-dialog-engines.adoc)

# #### 

# In[ ]:


text = "Vlad Snisar, Ruslan Borisov build ConvoHub w/ spaCy."
nlp(text).ents


# #### 

# In[ ]:


text = "Vlad snisar, ruslan borisov build convoHub w/ spaCy"
nlp(text).ents


# #### 

# In[ ]:


text = "Vlad snisar ruslan borisov convoHub spaCy"
nlp(text).ents


# #### 

# In[ ]:


def extract_proper_nouns(
        context, key="user_text",  # <1>
        pos="PROPN", ent_type=None):  # <2>
    doc = nlp(context.get(key, ''))  # <3>
    names = []
    i = 0
    while i < len(doc):
        tok = doc[i]
        ent = []
        if ((pos is None or tok.pos_ == pos)
                and (ent_type is None or tok.ent_type_ != ent_type)):
            for k, t in enumerate(doc[i:]):
                if not ((pos is None or t.pos_ == pos)
                    and (ent_type is None or t.ent_type_
                        != ent_type)):
                    break
                ent.append(t.text)
            names.append(" ".join(ent))
        i += len(ent) + 1
    return {'proper_nouns': names}


# #### 

# In[ ]:


text = 'Ruslan Borisov and Vlad Snisar rebuilt ConvoHub.'


# #### 

# In[ ]:


pip install mathtext
from mathtext.predict_intent import predict_intents_list
predict_intents_list('you are mean forty 2')  # <1>


# #### 

# In[ ]:


predict_intents_list('you are jerk infinity')


# #### 

# In[ ]:


def generate_question_data(start, stop, step, question_num=None):
    """ Generate list of possible questions with their contexts """
    seq = seq2str(start, stop, step)
    templates = [
        f"Let's practice counting {seq2str(start, stop, step)}... " \
        + f"What is the next number in the sequence after {stop}?",
        f"What number comes {step} after {stop}?\n{seq}",
        f"We're counting by {step}s. " \
        + f"What number is 1 after {stop}?\n{seq}",
        f"What is {step} number up from {stop}?\n{seq}",
        f"If we count up {step} from {stop}, " \
        + f"what number is next?\n{seq}",
    ]
    questions = []
    for quest in templates:
        questions.append({
             "question": quest,
             "answer": stop + step,
             "start": start,
             "stop": stop,
             "step": step,
             })
    return questions[question_num]


# #### 

# In[ ]:


from langchain.llms import Replicate
os.environ["REPLICATE_API_TOKEN"] = '<your_API_key_here>'
llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:" +
    "df7690",  # <1>
    input={
        "temperature": 0.5,
        "max_length": 100,
        "top_p": 1,
    })


# #### 

# In[ ]:


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
template = """
    This is a conversation between a math tutor 
    chatbot Rori and a user who might be a student 
    in Africa or a parent. 

    Human says: {message}
    Chatbot responds:
    """
prompt = PromptTemplate(
    input_variables = ["message"],  # <1>
    template=template)       
chain = LLMChain(
    llm=llm, verbose=True, prompt=prompt  # <2>
    )


# #### 

# In[ ]:


chain.predict(message="Hi Bot! My name is Maria.")


# #### 

# In[ ]:


chain.predict(message="What is my name?")


# #### 

# In[ ]:


template = """
    This is a conversation between a math tutor chatbot
    Rori and a user who might be a student in Africa or a parent. 
    The chatbot introduces itself and asks if it's talking to a
    student or to a parent. 
    If the user is a parent, Rori asks the parent for 
    permission for the child to use Rori over Whatsapp. 
    If the user is a student, Rori asks the student to
     call their parents. 
    If the parent agrees, Rori thanks them and asks to give the phone to the student. 
    Provide the tutor's next response based on the conversation history.

    {chat_history}
    Parent: {message}
    Tutor:"""
onboarding_prompt = PromptTemplate(
    input_variables = ["chat_history", "message"],
    template=template)


# #### 

# In[ ]:


memory = ConversationBufferMemory(
    memory_key='chat_history')  # <1>


# #### 

# In[ ]:


onboarding_chain = ConversationChain(
    llm=llm,
    memory = ConversationBufferMemory
    )
onboarding_chain.prompt = onboarding_prompt
onboarding_chain.predict(message="Hello")


# #### 

# In[ ]:


onboarding_chain.predict(message="I'm a parent")


# #### 

# In[ ]:


onboarding_pt = """


# #### 

# In[ ]:


class MathConversation():
    def __init__(self, llm, prompt_string):
        self.llm = llm
        self.memory = \
            ConversationBufferMemory(
                memory_key='history',
                ai_prefix='tutor',
                human_prefix="user")
        self.convo_chain = ConversationChain(
            llm=llm, memory=self.memory)
        self.convo_chain.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=prompt_string)

    def answer(self, user_input):
        return self.convo_chain.predict(input=user_input)


# #### 

# In[ ]:


onboarding_convo = MathConversation(llm, onboarding_pt)


# #### 

# In[ ]:


onboarding_convo.answer("I am a parent")


# #### 

# In[ ]:


onboarding_convo.answer("Yes, I agree")


# #### 

# In[ ]:


math_quiz_pt = """
You are a math teacher that's teaching math to a third-grade
student. Prompt the student to complete number sequences
from the following list and compare their answer with the
last number in the sequence:
  - 9,10,11,12
  - 38,39,40,41
  - 2,4,6,8
  - 1,5,9,13
  {history}
  student:{input}
  tutor:"""


# #### 

# In[ ]:


math_convo = MathConversation(llm, math_quiz_pt)
math_convo.answer("Let's start!")


# #### 

# In[ ]:


math_convo.answer("12")

